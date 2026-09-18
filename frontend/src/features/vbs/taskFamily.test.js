import { taskFamily } from './taskFamily';

describe('taskFamily', () => {
  test.each([
    [{ taskGroup: 'AVS', taskType: 'VIDEO SEARCH' }, 'AVS'],
    [{ taskGroup: 'misc', taskType: 'AD-HOC VIDEO SEARCH' }, 'AVS'],
    [{ taskGroup: 'misc', taskType: 'ADHOC VIDEO SEARCH' }, 'AVS'],
    [{ taskGroup: 'KIS', taskType: 'KNOWN ITEM SEARCH' }, 'KIS'],
    [{ taskGroup: 'TRAKE', taskType: 'TRAKE' }, 'TRAKE'],
    [{ taskGroup: 'QA', taskType: 'VQA' }, 'VQA'],
    [{ taskGroup: 'misc', taskType: 'misc', taskName: 'QA_01_CarColor' }, 'VQA'],
    [{ taskGroup: 'misc', taskType: 'misc', taskName: 'VQA_Test' }, 'VQA'],
    [{ taskGroup: 'misc', taskType: 'something else' }, 'OTHER'],
  ])('classifies %j as %s', (task, expected) => {
    expect(taskFamily(task)).toBe(expected);
  });
});
