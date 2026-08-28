/**
 * Create a small client-side ZIP archive for downloadable CSV submissions.
 *
 * The archive intentionally uses the ZIP "store" method instead of a
 * compression dependency: CSV files are small, and keeping the writer local
 * makes the submission workflow work offline in the browser.
 */

const encodeText = (value) => {
  if (typeof TextEncoder !== 'undefined') {
    return new TextEncoder().encode(value);
  }

  const encoded = encodeURIComponent(value);
  const bytes = [];
  for (let index = 0; index < encoded.length; index += 1) {
    if (encoded[index] === '%') {
      bytes.push(parseInt(encoded.slice(index + 1, index + 3), 16));
      index += 2;
    } else {
      bytes.push(encoded.charCodeAt(index));
    }
  }
  return Uint8Array.from(bytes);
};

const writeUint16 = (target, offset, value) => {
  target[offset] = value & 0xff;
  target[offset + 1] = (value >>> 8) & 0xff;
};

const writeUint32 = (target, offset, value) => {
  target[offset] = value & 0xff;
  target[offset + 1] = (value >>> 8) & 0xff;
  target[offset + 2] = (value >>> 16) & 0xff;
  target[offset + 3] = (value >>> 24) & 0xff;
};

const crc32 = (bytes) => {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
};

const concatBytes = (parts) => {
  const totalLength = parts.reduce((sum, part) => sum + part.length, 0);
  const output = new Uint8Array(totalLength);
  let offset = 0;
  parts.forEach((part) => {
    output.set(part, offset);
    offset += part.length;
  });
  return output;
};

export const getNonEmptyCsvFiles = (files) => (
  files.filter((file) => typeof file.content === 'string' && file.content.trim())
);

export const createCsvZip = (files) => {
  const entries = getNonEmptyCsvFiles(files).map((file) => {
    const name = encodeText(file.name);
    const content = encodeText(file.content);
    return {
      name,
      content,
      checksum: crc32(content),
    };
  });

  const localParts = [];
  const centralParts = [];
  let localOffset = 0;

  entries.forEach((entry) => {
    const localHeader = new Uint8Array(30 + entry.name.length);
    writeUint32(localHeader, 0, 0x04034b50);
    writeUint16(localHeader, 4, 20);
    writeUint16(localHeader, 6, 0x0800);
    writeUint16(localHeader, 8, 0);
    writeUint32(localHeader, 14, entry.checksum);
    writeUint32(localHeader, 18, entry.content.length);
    writeUint32(localHeader, 22, entry.content.length);
    writeUint16(localHeader, 26, entry.name.length);
    localHeader.set(entry.name, 30);
    localParts.push(localHeader, entry.content);

    const centralHeader = new Uint8Array(46 + entry.name.length);
    writeUint32(centralHeader, 0, 0x02014b50);
    writeUint16(centralHeader, 4, 20);
    writeUint16(centralHeader, 6, 20);
    writeUint16(centralHeader, 8, 0x0800);
    writeUint16(centralHeader, 10, 0);
    writeUint32(centralHeader, 16, entry.checksum);
    writeUint32(centralHeader, 20, entry.content.length);
    writeUint32(centralHeader, 24, entry.content.length);
    writeUint16(centralHeader, 28, entry.name.length);
    writeUint32(centralHeader, 42, localOffset);
    centralHeader.set(entry.name, 46);
    centralParts.push(centralHeader);

    localOffset += localHeader.length + entry.content.length;
  });

  const localData = concatBytes(localParts);
  const centralData = concatBytes(centralParts);
  const endRecord = new Uint8Array(22);
  writeUint32(endRecord, 0, 0x06054b50);
  writeUint16(endRecord, 8, entries.length);
  writeUint16(endRecord, 10, entries.length);
  writeUint32(endRecord, 12, centralData.length);
  writeUint32(endRecord, 16, localData.length);

  return new Blob([localData, centralData, endRecord], {
    type: 'application/zip',
  });
};

export const downloadCsvArchive = (
  files,
  { filename = 'submissions.zip', documentRef = document, urlApi = URL } = {},
) => {
  const nonEmptyFiles = getNonEmptyCsvFiles(files);
  if (nonEmptyFiles.length === 0) return false;

  const blob = createCsvZip(nonEmptyFiles);
  const url = urlApi.createObjectURL(blob);
  const anchor = documentRef.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  urlApi.revokeObjectURL(url);
  return true;
};
