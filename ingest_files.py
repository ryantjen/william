from __future__ import annotations
import hashlib
import io
from typing import Tuple, List, Dict, Any

import pandas as pd
from pypdf import PdfReader

from ingest import chunk_text

def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def ingest_txt(data: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Ingest plain text file from bytes."""
    text = data.decode("utf-8", errors="ignore")
    chunks = chunk_text(text)
    metas = [{"type": "txt_chunk", "name": filename, "chunk": i} for i in range(len(chunks))]
    return chunks, metas

def ingest_csv(data: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]], pd.DataFrame]:
    """
    Ingest CSV file with intelligent table understanding.
    Creates a dataset summary + column-aware chunks.
    """
    bio = io.BytesIO(data)
    df = pd.read_csv(bio)
    
    chunks = []
    metas = []
    
    # 1. Create dataset summary (always first chunk)
    summary = generate_dataset_summary(df, filename)
    chunks.append(summary)
    metas.append({
        "type": "csv_summary",
        "name": filename,
        "chunk": 0,
        "rows": len(df),
        "columns": len(df.columns)
    })
    
    # 2. Create column-by-column analysis for important columns
    for col_idx, col in enumerate(df.columns):
        col_analysis = generate_column_analysis(df, col)
        if col_analysis:
            chunks.append(col_analysis)
            metas.append({
                "type": "csv_column",
                "name": filename,
                "column": col,
                "chunk": col_idx + 1
            })
    
    # 3. Also chunk the raw data for semantic search (but smaller chunks)
    csv_text = df.head(100).to_csv(index=False)  # First 100 rows as sample
    data_chunks = chunk_text(csv_text, max_chars=2000, overlap=100)
    for i, ch in enumerate(data_chunks):
        chunks.append(f"**Sample data from {filename}:**\n\n{ch}")
        metas.append({
            "type": "csv_chunk",
            "name": filename,
            "chunk": len(chunks) - 1
        })
    
    return chunks, metas, df


def generate_dataset_summary(df: pd.DataFrame, filename: str) -> str:
    """Generate a comprehensive summary of the dataset."""
    lines = [
        f"**Dataset Summary: {filename}**\n",
        f"- **Rows:** {len(df):,}",
        f"- **Columns:** {len(df.columns)}",
        f"- **Memory usage:** {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n",
        "**Columns:**"
    ]
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
        
        if dtype in ['int64', 'float64']:
            # Numeric column
            lines.append(f"- `{col}` ({dtype}): min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}, nulls={null_pct:.1f}%")
        elif dtype == 'object':
            # Categorical/string column
            unique = df[col].nunique()
            sample_vals = df[col].dropna().head(3).tolist()
            sample_str = ", ".join([f'"{v}"' for v in sample_vals])
            lines.append(f"- `{col}` (text): {unique} unique values, samples: {sample_str}")
        else:
            lines.append(f"- `{col}` ({dtype}): {df[col].nunique()} unique values")
    
    # Add correlations for numeric columns if applicable
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) >= 2:
        lines.append("\n**Key correlations:**")
        try:
            corr = df[numeric_cols].corr()
            # Find top correlations (excluding self-correlation)
            pairs = []
            for i, c1 in enumerate(numeric_cols):
                for c2 in numeric_cols[i+1:]:
                    pairs.append((c1, c2, abs(corr.loc[c1, c2])))
            pairs.sort(key=lambda x: x[2], reverse=True)
            for c1, c2, r in pairs[:3]:  # Top 3 correlations
                if r > 0.3:  # Only show meaningful correlations
                    lines.append(f"- `{c1}` ↔ `{c2}`: r={r:.2f}")
        except Exception:
            pass
    
    return "\n".join(lines)


def generate_column_analysis(df: pd.DataFrame, col: str) -> str | None:
    """Generate detailed analysis for a single column."""
    if df[col].isnull().all():
        return None
    
    dtype = str(df[col].dtype)
    lines = [f"**Column Analysis: `{col}`**\n"]
    
    if dtype in ['int64', 'float64']:
        # Numeric column - detailed stats
        lines.extend([
            f"- Type: Numeric ({dtype})",
            f"- Count: {df[col].count():,} non-null values",
            f"- Mean: {df[col].mean():.4f}",
            f"- Std Dev: {df[col].std():.4f}",
            f"- Min: {df[col].min():.4f}",
            f"- 25%: {df[col].quantile(0.25):.4f}",
            f"- Median: {df[col].median():.4f}",
            f"- 75%: {df[col].quantile(0.75):.4f}",
            f"- Max: {df[col].max():.4f}",
        ])
    elif dtype == 'object':
        # Text/categorical column
        value_counts = df[col].value_counts().head(10)
        lines.extend([
            f"- Type: Text/Categorical",
            f"- Unique values: {df[col].nunique()}",
            f"- Most common values:"
        ])
        for val, count in value_counts.items():
            pct = count / len(df) * 100
            lines.append(f"  - \"{val}\": {count:,} ({pct:.1f}%)")
    else:
        # Other types (datetime, etc.)
        lines.extend([
            f"- Type: {dtype}",
            f"- Unique values: {df[col].nunique()}",
        ])
    
    return "\n".join(lines)

def ingest_pdf(data: bytes, filename: str, max_pages: int = 60) -> Tuple[List[str], List[Dict[str, Any]], int]:
    """Ingest text-based PDF file from bytes."""
    pdf_stream = io.BytesIO(data)
    reader = PdfReader(pdf_stream)
    total = min(len(reader.pages), max_pages)

    all_chunks = []
    all_metas = []
    pages_with_text = 0

    for page_idx in range(total):
        page = reader.pages[page_idx]
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        pages_with_text += 1

        chunks = chunk_text(page_text, max_chars=3000, overlap=200)
        for j, ch in enumerate(chunks):
            all_chunks.append(ch)
            all_metas.append({
                "type": "pdf_chunk",
                "name": filename,
                "page": page_idx + 1,
                "chunk": j
            })

    return all_chunks, all_metas, pages_with_text
