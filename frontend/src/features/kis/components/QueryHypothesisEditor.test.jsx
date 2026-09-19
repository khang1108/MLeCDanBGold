import React from 'react';
import { render, screen, fireEvent, within } from '@testing-library/react';
import QueryHypothesisEditor from './QueryHypothesisEditor';

describe('QueryHypothesisEditor component', () => {
  const baseIntent = {
    revision: 1,
    query_text: 'người đàn ông đi vào quán cà phê',
    language: 'vi',
    events: [
      {
        id: 'E1',
        text: 'người đàn ông đi vào',
        origin: 'source',
        source_provenance: { start_char: 0, end_char: 20 },
      },
      {
        id: 'E2',
        text: 'quán cà phê',
        origin: 'source',
        source_provenance: { start_char: 21, end_char: 32 },
      },
    ],
  };

  test('renders canonical query, revision badge, and events', () => {
    render(
      <QueryHypothesisEditor
        intent={baseIntent}
        canUndo={false}
      />
    );

    expect(screen.getByText(/Rev 1/i)).toBeInTheDocument();
    expect(screen.getByText(/VI/i)).toBeInTheDocument();
    expect(screen.getByText(/người đàn ông đi vào quán cà phê/i)).toBeInTheDocument();
    expect(screen.getByText('người đàn ông đi vào')).toBeInTheDocument();
    expect(screen.getByText('quán cà phê')).toBeInTheDocument();
  });

  test('apply split commits preview but does not call search automatically', async () => {
    const props = {
      intent: baseIntent,
      preview: {
        base_revision: 1,
        intent: {
          revision: 2,
          events: [
            { id: 'E1', text: 'người đàn ông' },
            { id: 'E2', text: 'đi vào' },
            { id: 'E3', text: 'quán cà phê' },
          ],
        },
      },
      onCommit: jest.fn(),
      onSearch: jest.fn(),
      onCancelPreview: jest.fn(),
    };

    render(<QueryHypothesisEditor {...props} />);

    const applyBtn = screen.getByRole('button', { name: /apply/i });
    expect(applyBtn).toBeInTheDocument();

    fireEvent.click(applyBtn);

    expect(props.onCommit).toHaveBeenCalledTimes(1);
    expect(props.onSearch).not.toHaveBeenCalled();
  });

  test('cancel preview triggers onCancelPreview', async () => {
    const props = {
      intent: baseIntent,
      preview: {
        base_revision: 1,
        intent: { revision: 2, events: [] },
      },
      onCancelPreview: jest.fn(),
    };

    render(<QueryHypothesisEditor {...props} />);

    const cancelBtn = screen.getByRole('button', { name: /cancel/i });
    fireEvent.click(cancelBtn);

    expect(props.onCancelPreview).toHaveBeenCalledTimes(1);
  });

  test('renders stale results notice when isResultsStale is true', async () => {
    const onSearch = jest.fn();

    render(
      <QueryHypothesisEditor
        intent={baseIntent}
        isResultsStale={true}
        onSearch={onSearch}
      />
    );

    expect(screen.getByTestId('stale-results-notice')).toBeInTheDocument();
    const searchBtn = screen.getByRole('button', { name: /^search$/i });
    fireEvent.click(searchBtn);
    expect(onSearch).toHaveBeenCalledTimes(1);
  });

  test.each([
    ['edit', () => {
      fireEvent.click(within(screen.getByTestId('hypothesis-event-E1')).getByTitle('Edit event text'));
      const input = screen.getByDisplayValue('người đàn ông đi vào');
      fireEvent.change(input, { target: { value: 'người đàn ông bước vào' } });
      fireEvent.click(screen.getByRole('button', { name: /preview edit/i }));
    }, { type: 'edit', event_id: 'E1', text: 'người đàn ông bước vào' }],
    ['split', () => {
      fireEvent.click(within(screen.getByTestId('hypothesis-event-E1')).getByTitle('Split this event into two sequential events'));
      fireEvent.click(screen.getByRole('button', { name: /preview split/i }));
    }, expect.objectContaining({ type: 'split', event_id: 'E1', image_assignments: {} })],
    ['merge', () => {
      fireEvent.click(within(screen.getByTestId('hypothesis-event-E1')).getByTitle('Merge with E2'));
    }, { type: 'merge', left_event_id: 'E1', right_event_id: 'E2' }],
    ['reorder', () => {
      fireEvent.click(within(screen.getByTestId('hypothesis-event-E1')).getByRole('button', { name: 'Move event later' }));
    }, { type: 'reorder', event_ids: ['E2', 'E1'] }],
    ['add', () => {
      fireEvent.click(screen.getByRole('button', { name: /add event at start/i }));
      fireEvent.change(screen.getByPlaceholderText('New event description...'), {
        target: { value: 'a new clue' },
      });
      fireEvent.click(screen.getByRole('button', { name: /^add$/i }));
    }, { type: 'add', position: 0, text: 'a new clue', images: [] }],
  ])('emits the backend %s discriminator for direct editor actions', (_name, perform, expected) => {
    const onPreviewAction = jest.fn();
    render(<QueryHypothesisEditor intent={baseIntent} onPreviewAction={onPreviewAction} />);

    perform();

    expect(onPreviewAction).toHaveBeenCalledWith(expected);
  });

  test('requires explicit left/right assignment for every split image and sends it', () => {
    const onPreviewAction = jest.fn();
    const intent = {
      ...baseIntent,
      events: [{
        ...baseIntent.events[0],
        text: 'person enters room',
        images: [
          { asset_id: 'asset-left' },
          { asset_id: 'asset-right' },
        ],
      }],
    };

    render(<QueryHypothesisEditor intent={intent} onPreviewAction={onPreviewAction} />);
    fireEvent.click(within(screen.getByTestId('hypothesis-event-E1')).getByTitle('Split this event into two sequential events'));

    const previewSplit = screen.getByRole('button', { name: /preview split/i });
    expect(previewSplit).toBeDisabled();
    fireEvent.click(screen.getByLabelText('Assign asset-left to left child'));
    fireEvent.click(screen.getByLabelText('Assign asset-right to right child'));
    expect(previewSplit).not.toBeDisabled();
    fireEvent.click(previewSplit);

    expect(onPreviewAction).toHaveBeenCalledWith(expect.objectContaining({
      type: 'split',
      image_assignments: {
        'asset-left': ['left'],
        'asset-right': ['right'],
      },
    }));
  });
});
