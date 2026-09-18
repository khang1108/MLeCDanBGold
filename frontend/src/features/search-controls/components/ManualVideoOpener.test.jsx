import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ManualVideoOpener from './ManualVideoOpener';
import * as framesApi from '../../../api/frames';

describe('ManualVideoOpener', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('validates inputs before submission', () => {
    render(<ManualVideoOpener onOpenFrame={jest.fn()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Open in viewer' }));
    expect(screen.getByRole('alert').textContent).toContain('Enter a video_id.');

    fireEvent.change(screen.getByLabelText('video_id'), { target: { value: 'V001' } });
    fireEvent.click(screen.getByRole('button', { name: 'Open in viewer' }));
    expect(screen.getByRole('alert').textContent).toContain('timestamp_ms must be a non-negative base-10 integer.');

    fireEvent.change(screen.getByLabelText('timestamp_ms'), { target: { value: '-5' } });
    fireEvent.click(screen.getByRole('button', { name: 'Open in viewer' }));
    expect(screen.getByRole('alert').textContent).toContain('timestamp_ms must be a non-negative base-10 integer.');
  });

  test('resolves frame and calls onOpenFrame with modal data', async () => {
    const handleOpenFrame = jest.fn();
    const mockFrame = {
      frame_id: 'f-1',
      video_id: 'V001',
      frame_idx: 120,
      timestamp_ms: 12000,
      requested_timestamp_ms: 12000,
      metadata: {},
    };
    jest.spyOn(framesApi, 'resolveFrameAtTimestamp').mockResolvedValue(mockFrame);

    render(<ManualVideoOpener onOpenFrame={handleOpenFrame} />);

    fireEvent.change(screen.getByLabelText('video_id'), { target: { value: 'V001' } });
    fireEvent.change(screen.getByLabelText('timestamp_ms'), { target: { value: '12000' } });
    fireEvent.click(screen.getByRole('button', { name: 'Open in viewer' }));

    await waitFor(() => {
      expect(handleOpenFrame).toHaveBeenCalledWith({
        frame: mockFrame,
        initialTimestampMs: 12000,
      });
    });
  });

  test('displays error message when resolve fails', async () => {
    jest.spyOn(framesApi, 'resolveFrameAtTimestamp').mockRejectedValue(new Error('Video not found'));

    render(<ManualVideoOpener onOpenFrame={jest.fn()} />);

    fireEvent.change(screen.getByLabelText('video_id'), { target: { value: 'Unknown' } });
    fireEvent.change(screen.getByLabelText('timestamp_ms'), { target: { value: '1000' } });
    fireEvent.click(screen.getByRole('button', { name: 'Open in viewer' }));

    await waitFor(() => {
      expect(screen.getByRole('alert').textContent).toContain('Video not found');
    });
  });
});
