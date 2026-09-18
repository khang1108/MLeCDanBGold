/**
 * Task family classifier matching backend DRES task semantics.
 *
 * Distinguishes AVS, VQA, KIS, TRAKE, and OTHER based on taskGroup and taskType.
 */

export const taskFamily = (task) => {
  if (!task || typeof task !== 'object') {
    return 'OTHER';
  }

  const group = typeof task.taskGroup === 'string' ? task.taskGroup.trim().toUpperCase() : '';
  const type = typeof task.taskType === 'string' ? task.taskType.trim().toUpperCase() : '';
  const name = typeof task.taskName === 'string' ? task.taskName.trim().toUpperCase() : '';

  if (
    group.includes('QA') ||
    type.includes('QA') ||
    group.includes('VQA') ||
    type.includes('VQA') ||
    name.includes('VQA') ||
    name.startsWith('QA') ||
    name.includes('_QA') ||
    name.includes('-QA')
  ) {
    return 'VQA';
  }

  if (
    group.includes('AVS') ||
    type.includes('AVS') ||
    type.includes('AD-HOC') ||
    type.includes('ADHOC') ||
    group.includes('AD-HOC') ||
    group.includes('ADHOC')
  ) {
    return 'AVS';
  }

  if (group.includes('KIS') || type.includes('KIS') || type.includes('KNOWN')) {
    return 'KIS';
  }

  if (group.includes('TRAKE') || type.includes('TRAKE')) {
    return 'TRAKE';
  }

  return 'OTHER';
};
