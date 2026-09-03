import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import SubmissionFileModal from './SubmissionFileModal';

const baseProps = {
  mode: 'editor',
  files: [{ name: 'a.csv', content: 'old', is_validated: false, revision: 1 }],
  editorFile: { name: 'a.csv', content: 'old', is_validated: false, revision: 1 },
  draft: 'old',
  onDraftChange: jest.fn(),
  onSave: jest.fn(),
  onValidate: jest.fn(),
  onDelete: jest.fn(),
  onClose: jest.fn(),
};

test('plain Enter saves and Shift+Enter remains a newline action', () => {
  render(<SubmissionFileModal {...baseProps} />);
  const editor = screen.getByRole('textbox', { name: /edit a\.csv content/i });
  fireEvent.keyDown(editor, { key: 'Enter', code: 'Enter' });
  expect(baseProps.onSave).toHaveBeenCalledTimes(1);
  fireEvent.keyDown(editor, { key: 'Enter', code: 'Enter', shiftKey: true });
  expect(baseProps.onSave).toHaveBeenCalledTimes(1);
});

test('Escape closes without invoking save', () => {
  render(<SubmissionFileModal {...baseProps} />);
  fireEvent.keyDown(screen.getByRole('textbox', { name: /edit a\.csv content/i }), { key: 'Escape' });
  expect(baseProps.onSave).not.toHaveBeenCalled();
  expect(baseProps.onClose).toHaveBeenCalledTimes(1);
});

test('close controls remain available while a save confirmation is pending', () => {
  const onClose = jest.fn();
  render(<SubmissionFileModal {...baseProps} isMutating onClose={onClose} />);

  const closeButton = screen.getByRole('button', { name: 'Close' });
  const cancelButton = screen.getByRole('button', { name: 'Cancel' });
  expect(closeButton.disabled).toBe(false);
  expect(cancelButton.disabled).toBe(false);

  fireEvent.click(cancelButton);
  expect(onClose).toHaveBeenCalledTimes(1);
});
