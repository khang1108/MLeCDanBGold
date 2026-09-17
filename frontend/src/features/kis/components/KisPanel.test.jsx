import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import KisPanel from './KisPanel';
import { kisImageAssetUrl } from '../../../api/kis';

const STATE_WITH_E1_E2 = {
  draft: '',
  revision: 2,
  currentIntent: {
    revision: 2,
    query_text: 'woman enters, then chef appears',
    entities: [
      { id: 'X1', kind: 'person', description: 'woman' },
      { id: 'X2', kind: 'person', description: 'chef' },
    ],
    events: [
      {
        id: 'E1',
        text: 'woman enters',
        images: [{ asset_id: 'ast_e1_img', file_name: 'ref1.jpg' }],
        bindings: [],
      },
      { id: 'E2', text: 'chef appears', images: [], bindings: [] },
    ],
    temporal_edges: [{ source: 'E1', target: 'E2', relation: 'before' }],
  },
  stagedImages: {},
  pendingOperation: null,
  isSearching: false,
  error: null,
  mode: 'live',
};

describe('KisPanel unified multimodal component', () => {
  test('renders committed multimodal events and prefills the next event', () => {
    const onDraftChange = jest.fn();
    render(<KisPanel sessionState={STATE_WITH_E1_E2} onDraftChange={onDraftChange} />);
    expect(screen.getByText('E1')).toBeTruthy();
    expect(screen.getByText('E2')).toBeTruthy();
    expect(screen.getByText('woman enters')).toBeTruthy();
    expect(screen.getByText('chef appears')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: /add event/i }));
    expect(onDraftChange).toHaveBeenCalledWith('E3: ');
  });

  test('asks for an event target when an image is attached without E# scope', async () => {
    const file = new File(['png-bytes'], 'chef.png', { type: 'image/png' });
    const onAttachImage = jest.fn();
    render(
      <KisPanel
        sessionState={STATE_WITH_E1_E2}
        onDraftChange={jest.fn()}
        onAttachImage={onAttachImage}
      />,
    );
    fireEvent.change(screen.getByLabelText(/attach image/i), { target: { files: [file] } });
    expect(screen.getByText(/attach image to/i)).toBeTruthy();
    expect(screen.getByRole('button', { name: 'E1' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'E2' })).toBeTruthy();
    expect(onAttachImage).not.toHaveBeenCalled();

    // Clicking E2 triggers onAttachImage
    fireEvent.click(screen.getByRole('button', { name: 'E2' }));
    expect(onAttachImage).toHaveBeenCalledWith(file, 'E2');
  });

  test('initial image-only targets E1 directly', () => {
    const file = new File(['png-bytes'], 'solo.png', { type: 'image/png' });
    const onAttachImage = jest.fn();
    const initialState = {
      draft: '',
      revision: 0,
      currentIntent: null,
      stagedImages: {},
      isSearching: false,
      error: null,
      mode: 'live',
    };
    render(<KisPanel sessionState={initialState} onAttachImage={onAttachImage} />);
    fireEvent.change(screen.getByLabelText(/attach image/i), { target: { files: [file] } });
    expect(onAttachImage).toHaveBeenCalledWith(file, 'E1');
  });

  test('initial text+image targets one E1', () => {
    const file = new File(['png-bytes'], 'kitchen.png', { type: 'image/png' });
    const onAttachImage = jest.fn();
    const initialState = {
      draft: 'woman enters the kitchen',
      revision: 0,
      currentIntent: null,
      stagedImages: {},
      isSearching: false,
      error: null,
      mode: 'live',
    };
    render(<KisPanel sessionState={initialState} onAttachImage={onAttachImage} />);
    fireEvent.change(screen.getByLabelText(/attach image/i), { target: { files: [file] } });
    expect(onAttachImage).toHaveBeenCalledWith(file, 'E1');
  });

  test('E2: scoped draft + pasted image immediately targets E2', () => {
    const file = new File(['png-bytes'], 'pasted.png', { type: 'image/png' });
    const onAttachImage = jest.fn();
    const stateWithE2Draft = {
      ...STATE_WITH_E1_E2,
      draft: 'E2: chef wears black hat',
    };
    render(<KisPanel sessionState={stateWithE2Draft} onAttachImage={onAttachImage} />);

    const textarea = screen.getByPlaceholderText('Search or add another clue…');
    fireEvent.paste(textarea, {
      clipboardData: {
        items: [{ type: 'image/png', getAsFile: () => file }],
      },
    });

    expect(onAttachImage).toHaveBeenCalledWith(file, 'E2');
  });

  test('renders persisted thumbnail URL pointing to kisImageAssetUrl(asset_id)', () => {
    render(<KisPanel sessionState={STATE_WITH_E1_E2} />);
    const img = screen.getByAltText('ref1.jpg');
    expect(img).toBeTruthy();
    expect(img.getAttribute('src')).toBe(kisImageAssetUrl('ast_e1_img'));
  });

  test('calls onRemoveImage when remove button on event image is clicked', () => {
    const onRemoveImage = jest.fn();
    render(<KisPanel sessionState={STATE_WITH_E1_E2} onRemoveImage={onRemoveImage} />);
    const removeBtn = screen.getByRole('button', { name: /remove image ref1.jpg/i });
    fireEvent.click(removeBtn);
    expect(onRemoveImage).toHaveBeenCalledWith('E1', 'ast_e1_img');
  });

  test('displays dynamic submit label Rewrite for /llm-rewrite', () => {
    const rewriteState = {
      ...STATE_WITH_E1_E2,
      draft: '/llm-rewrite\nrewrite everything for restaurant setting',
    };
    render(<KisPanel sessionState={rewriteState} />);
    expect(screen.getByRole('button', { name: 'Rewrite' })).toBeTruthy();
  });

  test('displays dynamic submit label Update for patch_events', () => {
    const patchState = {
      ...STATE_WITH_E1_E2,
      draft: 'E2: chef stirs soup',
    };
    render(<KisPanel sessionState={patchState} />);
    expect(screen.getByRole('button', { name: 'Update' })).toBeTruthy();
  });

  test('calls onReset when reset button is clicked', () => {
    const onReset = jest.fn();
    render(<KisPanel sessionState={STATE_WITH_E1_E2} onReset={onReset} />);
    fireEvent.click(screen.getByRole('button', { name: /new search/i }));
    expect(onReset).toHaveBeenCalled();
  });

  test('renders collapse button when onCollapse is provided and calls it on click', () => {
    const onCollapse = jest.fn();
    render(<KisPanel sessionState={STATE_WITH_E1_E2} onCollapse={onCollapse} />);
    const collapseBtn = screen.getByRole('button', { name: /collapse kis search panel/i });
    expect(collapseBtn).toBeTruthy();
    fireEvent.click(collapseBtn);
    expect(onCollapse).toHaveBeenCalledTimes(1);
  });
});

