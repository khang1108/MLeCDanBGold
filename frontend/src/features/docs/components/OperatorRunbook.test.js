import fs from 'fs';
import path from 'path';

const readme = fs.readFileSync(path.resolve(__dirname, '../../../../../README.md'), 'utf8');

test('archives the retired shared-workspace data before migration', () => {
  expect(readme).toMatch(/backup[^\n]*runtime\/workspace\.sqlite3/i);
  expect(readme).toMatch(/submission-file[^\n]*(retired|retirement)/i);
});

test('documents the VBS 2027 operator rehearsal and freeze checklist', () => {
  [
    'HCMAI_DRES_USERS_JSON',
    'HCMAI_DRES_MEDIA_ID_PREFIX_TO_STRIP',
    'mediaItemName',
    'each browser',
    'test DRES',
    'KIS',
    'VQA',
    'AVS',
    'timestamp_ms',
    'connected participant',
    'freeze',
  ].forEach((item) => expect(readme).toContain(item));
});
