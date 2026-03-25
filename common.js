function splitText(text, chunkSize = 600, overlap = 120) {
  if (!text || chunkSize <= 0) return [];

  const normalizedText = text.replace(/\r\n/g, "\n").trim();
  if (!normalizedText) return [];

  const chunks = [];
  let start = 0;

  while (start < normalizedText.length) {
    let end = Math.min(start + chunkSize, normalizedText.length);

    if (end < normalizedText.length) {
      const searchStart = Math.max(start, end - 80);
      const breakpoint = normalizedText.lastIndexOf("\n", end);

      if (breakpoint >= searchStart) {
        end = breakpoint;
      }
    }

    const chunk = normalizedText.slice(start, end).trim();
    if (chunk) {
      chunks.push(chunk);
    }

    if (end >= normalizedText.length) {
      break;
    }

    start = Math.max(end - overlap, start + 1);
  }

  return chunks;
}

function cosineSimilarity(a, b) {
  if (a.length !== b.length) {
    throw new Error("Vectors must have the same dimension");
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dot / (normA * normB);
}

export { splitText, cosineSimilarity };
