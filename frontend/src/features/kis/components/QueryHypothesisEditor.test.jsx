import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
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
});
