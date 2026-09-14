import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import AnswerCandidateDialog from './AnswerCandidateDialog';

test('Enter saves an editable FRAME draft with only frame coordinates', () => {
  const onSave = jest.fn();
  const onCancel = jest.fn();
  render(
    <AnswerCandidateDialog
      kind="FRAME"
      initialValue={{ kind: 'FRAME', videoId: 'V01', timestampMs: 12_345 }}
      onSave={onSave}
      onCancel={onCancel}
    />,
  );

  const timestamp = screen.getByLabelText('Timestamp (ms)');
  expect(timestamp.value).toBe('12345');
  fireEvent.keyDown(timestamp, { key: 'Enter', code: 'Enter' });

  expect(onSave).toHaveBeenCalledWith({ kind: 'FRAME', videoId: 'V01', timestampMs: 12_345 });
  expect(onCancel).not.toHaveBeenCalled();
});

test('Enter saves editable TEXT without frame fields', () => {
  const onSave = jest.fn();
  render(
    <AnswerCandidateDialog
      kind="TEXT"
      initialValue={{ kind: 'TEXT', text: 'A person enters the room' }}
      onSave={onSave}
      onCancel={jest.fn()}
    />,
  );

  const text = screen.getByLabelText('Text answer');
  expect(screen.queryByLabelText('Video ID')).toBeNull();
  expect(screen.queryByLabelText('Timestamp (ms)')).toBeNull();
  fireEvent.keyDown(text, { key: 'Enter', code: 'Enter' });

  expect(onSave).toHaveBeenCalledWith({ kind: 'TEXT', text: 'A person enters the room' });
});

test('Escape cancels without saving', () => {
  const onSave = jest.fn();
  const onCancel = jest.fn();
  render(
    <AnswerCandidateDialog
      kind="TEXT"
      initialValue={{ kind: 'TEXT', text: 'candidate' }}
      onSave={onSave}
      onCancel={onCancel}
    />,
  );

  fireEvent.keyDown(screen.getByLabelText('Text answer'), { key: 'Escape', code: 'Escape' });

  expect(onCancel).toHaveBeenCalledTimes(1);
  expect(onSave).not.toHaveBeenCalled();
});
