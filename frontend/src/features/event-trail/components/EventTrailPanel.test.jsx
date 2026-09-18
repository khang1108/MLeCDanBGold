import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import EventTrailPanel from './EventTrailPanel';

const mockEvents = [
  { id: 'E1', text: 'woman enters' },
  { id: 'E2', text: 'woman sits' },
  { id: 'E3', text: 'woman talks' },
];

const makeState = (overrides = {}) => ({
  session_id: 'ses_1',
  result_id: 'r_1',
  video_id: 'V01',
  kis_revision: 2,
  trail_revision: 1,
  status: 'active',
  path: [
    { event_id: 'E1', frame_id: 'f1', frame_idx: 10, timestamp_ms: 1000 },
    { event_id: 'E2', frame_id: 'f2', frame_idx: 20, timestamp_ms: 2000 },
    { event_id: 'E3', frame_id: 'f3', frame_idx: 30, timestamp_ms: 3000 },
  ],
  last_valid_path: [
    { event_id: 'E1', frame_id: 'f1', frame_idx: 10, timestamp_ms: 1000 },
    { event_id: 'E2', frame_id: 'f2', frame_idx: 20, timestamp_ms: 2000 },
    { event_id: 'E3', frame_id: 'f3', frame_idx: 30, timestamp_ms: 3000 },
  ],
  approved_event_ids: [],
  rejected_counts: { E1: 0, E2: 0, E3: 0 },
  window: null,
  submission_selection: null,
  transition: null,
  ...overrides,
});

describe('EventTrailPanel', () => {
  test('Step 1: renders all events, timestamps, and thumbnails, and allows event selection', () => {
    const onSelectEvent = jest.fn();

    render(
      <EventTrailPanel
        events={mockEvents}
        state={makeState()}
        selectedEventId="E1"
        onSelectEvent={onSelectEvent}
      />
    );

    expect(screen.getByRole('heading', { name: 'Hypothesis Explorer' })).toBeTruthy();
    expect(screen.getAllByText('woman enters').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('woman sits')).toBeTruthy();
    expect(screen.getByText('woman talks')).toBeTruthy();

    // Clicking E2 selects it
    const e2Item = screen.getByTestId('event-rail-item-E2');
    fireEvent.click(e2Item);
    expect(onSelectEvent).toHaveBeenCalledWith('E2');
  });

  test('Step 2: action buttons respect approval and exhaustion state', () => {
    const onApprove = jest.fn();
    const onDecline = jest.fn();
    const onUse = jest.fn();
    const onClearAnchor = jest.fn();

    // 1. Active state, E1 not approved
    const { rerender } = render(
      <EventTrailPanel
        events={mockEvents}
        state={makeState()}
        selectedEventId="E1"
        onApprove={onApprove}
        onDecline={onDecline}
        onUse={onUse}
      />
    );

    const keepBtn = screen.getByRole('button', { name: /^keep$/i });
    const rejectBtn = screen.getByRole('button', { name: /reject occurrence/i });
    const useBtn = screen.getByRole('button', { name: /use \(manual frame\)/i });

    expect(keepBtn.disabled).toBe(false);
    expect(rejectBtn.disabled).toBe(false);
    expect(useBtn.disabled).toBe(false);
    expect(screen.queryByRole('button', { name: /clear anchor/i })).toBeNull();

    fireEvent.click(keepBtn);
    expect(onApprove).toHaveBeenCalledWith('E1');

    fireEvent.click(rejectBtn);
    expect(onDecline).toHaveBeenCalledWith('E1');

    fireEvent.click(useBtn);
    expect(onUse).toHaveBeenCalledWith('E1');

    // 2. E1 is approved: approve and decline disabled, clear anchor visible
    rerender(
      <EventTrailPanel
        events={mockEvents}
        state={makeState({ approved_event_ids: ['E1'] })}
        selectedEventId="E1"
        onApprove={onApprove}
        onDecline={onDecline}
        onUse={onUse}
        onClearAnchor={onClearAnchor}
      />
    );

    expect(screen.getByRole('button', { name: /^keep$/i }).disabled).toBe(true);
    expect(screen.getByRole('button', { name: /reject occurrence/i }).disabled).toBe(true);
    const clearBtn = screen.getByRole('button', { name: /clear anchor/i });
    expect(clearBtn).toBeTruthy();
    fireEvent.click(clearBtn);
    expect(onClearAnchor).toHaveBeenCalledWith('E1');

    // 3. Exhausted state: actions disabled, submit disabled
    rerender(
      <EventTrailPanel
        events={mockEvents}
        state={makeState({ status: 'exhausted', path: null })}
        selectedEventId="E1"
        onApprove={onApprove}
        onDecline={onDecline}
        onUse={onUse}
      />
    );

    expect(screen.getByRole('button', { name: /^keep$/i }).disabled).toBe(true);
    expect(screen.getByRole('button', { name: /reject occurrence/i }).disabled).toBe(true);
    expect(screen.getByRole('button', { name: /use \(manual frame\)/i }).disabled).toBe(true);
    expect(screen.getByRole('button', { name: /^submit$/i }).disabled).toBe(true);
  });

  test('reject occurrence submits mode id', () => {
    const onRejectMode = jest.fn();
    const propsWithFocusedMode = {
      events: mockEvents,
      state: makeState(),
      selectedEventId: 'E2',
      focusedModeId: 'mode_current',
      onRejectMode,
    };
    render(<EventTrailPanel {...propsWithFocusedMode} />);
    const rejectBtn = screen.getByRole('button', { name: /reject occurrence/i });
    fireEvent.click(rejectBtn);
    expect(onRejectMode).toHaveBeenCalledWith('E2', 'mode_current');
  });

  test('renders HypothesisPathPreview and commits alternative with onUseAlternative', () => {
    const onUseAlternative = jest.fn();
    const onClearPreview = jest.fn();
    const previewAlt = {
      alternative_id: 'alt_99',
      event_id: 'E2',
      path: [
        { event_id: 'E1', frame_id: 'f1', frame_idx: 10, timestamp_ms: 1000 },
        { event_id: 'E2', frame_id: 'f2_new', frame_idx: 25, timestamp_ms: 2500 },
        { event_id: 'E3', frame_id: 'f3_new', frame_idx: 35, timestamp_ms: 3500 },
      ],
    };
    render(
      <EventTrailPanel
        events={mockEvents}
        state={makeState()}
        selectedEventId="E2"
        previewAlternative={previewAlt}
        onUseAlternative={onUseAlternative}
        onClearPreview={onClearPreview}
      />
    );

    expect(screen.getByTestId('hypothesis-path-preview')).toBeInTheDocument();
    expect(screen.getByTestId('adjusted-event')).toHaveTextContent(/adjusted to maintain order/i);

    const useAltBtns = screen.getAllByRole('button', { name: /use this occurrence/i });
    fireEvent.click(useAltBtns[0]);
    expect(onUseAlternative).toHaveBeenCalledWith('E2', 'alt_99');
  });

  test('displays stale query banner when currentQueryRevision differs from state.kis_revision', () => {
    render(
      <EventTrailPanel
        events={mockEvents}
        state={makeState({ kis_revision: 1 })}
        currentQueryRevision={2}
        selectedEventId="E1"
      />
    );

    expect(screen.getByTestId('trail-stale-query-notice')).toHaveTextContent(/based on query revision 1/i);
  });

  test('Step 3: renders diff transitions and exhaustion messaging', () => {
    const onUndo = jest.fn();
    const onBack = jest.fn();

    const transitionState = makeState({
      transition: {
        action_event_id: 'E2',
        direct_changed_event_ids: ['E2'],
        indirect_changed_event_ids: ['E1', 'E3'],
        candidate_diffs: [
          {
            event_id: 'E2',
            before_frame_id: 'f2_old',
            after_frame_id: 'f2',
            before_timestamp_ms: 1800,
            after_timestamp_ms: 2000,
          },
        ],
      },
    });

    const { rerender } = render(
      <EventTrailPanel
        events={mockEvents}
        state={transitionState}
        selectedEventId="E2"
        onUndo={onUndo}
        onBack={onBack}
      />
    );

    // Shows E2 as acted-on and other events updated
    expect(screen.getByText(/2 other events updated/i)).toBeTruthy();

    // Rerender exhausted
    rerender(
      <EventTrailPanel
        events={mockEvents}
        state={makeState({ status: 'exhausted', path: null })}
        selectedEventId="E1"
        onUndo={onUndo}
        onBack={onBack}
      />
    );

    expect(screen.getByText(/No valid path remains in this video/i)).toBeTruthy();

    // Undo and Exit remain clickable
    const undoBtn = screen.getByRole('button', { name: /^undo$/i });
    const exitBtn = screen.getByRole('button', { name: /^exit$/i });
    expect(undoBtn.disabled).toBe(false);
    expect(exitBtn.disabled).toBe(false);

    fireEvent.click(undoBtn);
    expect(onUndo).toHaveBeenCalledTimes(1);

    fireEvent.click(exitBtn);
    expect(onBack).toHaveBeenCalledTimes(1);
  });

  test('Step 4: secondary window controls collapse/expand and emit set_window / clear_window', () => {
    const onSetWindow = jest.fn();
    const onClearWindow = jest.fn();

    render(
      <EventTrailPanel
        events={mockEvents}
        state={makeState()}
        selectedEventId="E1"
        onSetWindow={onSetWindow}
        onClearWindow={onClearWindow}
      />
    );

    // Collapsed by default
    expect(screen.queryByLabelText(/start ms/i)).toBeNull();

    // Expand search range
    const toggleBtn = screen.getByRole('button', { name: /search range/i });
    fireEvent.click(toggleBtn);

    const startInput = screen.getByLabelText(/start ms/i);
    const endInput = screen.getByLabelText(/end ms/i);
    fireEvent.change(startInput, { target: { value: '10000' } });
    fireEvent.change(endInput, { target: { value: '40000' } });

    const applyBtn = screen.getByRole('button', { name: /apply range/i });
    fireEvent.click(applyBtn);
    expect(onSetWindow).toHaveBeenCalledWith({
      type: 'set_window',
      start_ms: 10000,
      end_ms: 40000,
    });

    const clearBtn = screen.getByRole('button', { name: /clear range/i });
    fireEvent.click(clearBtn);
    expect(onClearWindow).toHaveBeenCalledTimes(1);
  });

  test('Step 3 (Task 7): cycles selection through indirect_changed_event_ids when review button is clicked', () => {
    const onSelectEvent = jest.fn();
    const transitionState = makeState({
      transition: {
        action_event_id: 'E2',
        direct_changed_event_ids: ['E2'],
        indirect_changed_event_ids: ['E1', 'E3'],
        candidate_diffs: [],
      },
    });

    render(
      <EventTrailPanel
        events={mockEvents}
        state={transitionState}
        selectedEventId="E2"
        onSelectEvent={onSelectEvent}
      />
    );

    const cycleBtn = screen.getByRole('button', { name: /2 other events updated/i });
    expect(cycleBtn).toBeTruthy();

    fireEvent.click(cycleBtn);
    expect(onSelectEvent).toHaveBeenCalledWith('E1');
  });

  test('Exit button triggers onExitTrail', () => {
    const onExitTrail = jest.fn();
    render(
      <EventTrailPanel
        events={mockEvents}
        state={makeState()}
        selectedEventId="E1"
        onExitTrail={onExitTrail}
      />
    );

    const exitBtn = screen.getByRole('button', { name: /^exit$/i });
    expect(exitBtn).toBeTruthy();
    fireEvent.click(exitBtn);
    expect(onExitTrail).toHaveBeenCalledTimes(1);
  });
});

