import { createCsvZip, getNonEmptyCsvFiles } from './submissionArchive';

test('filters whitespace-only CSV files', () => {
  expect(getNonEmptyCsvFiles([
    { name: 'filled.csv', content: 'L21_V001,100' },
    { name: 'empty.csv', content: ' \n ' },
    { name: 'missing.csv' },
  ])).toEqual([
    { name: 'filled.csv', content: 'L21_V001,100' },
  ]);
});

test('creates a non-empty ZIP blob with the expected MIME type', () => {
  const blob = createCsvZip([
    { name: 'filled.csv', content: 'L21_V001,100' },
    { name: 'empty.csv', content: '' },
  ]);

  expect(blob.type).toBe('application/zip');
  expect(blob.size).toBeGreaterThan(0);
});
