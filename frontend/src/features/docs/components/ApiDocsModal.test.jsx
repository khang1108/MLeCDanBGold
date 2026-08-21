import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import ApiDocsModal from './ApiDocsModal';

describe('ApiDocsModal component', () => {
  test('does not render when isOpen is false', () => {
    const { container } = render(<ApiDocsModal isOpen={false} onClose={jest.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  test('renders Swagger tab by default when isOpen is true', () => {
    render(<ApiDocsModal isOpen={true} onClose={jest.fn()} />);
    expect(screen.getByText('Interactive API Documentation')).toBeTruthy();
    expect(screen.getByTitle('FastAPI Swagger UI')).toBeTruthy();
    expect(screen.getByText(/Swagger \/docs/)).toBeTruthy();
  });

  test('switches to endpoints quick reference tab and displays endpoint cards', () => {
    render(<ApiDocsModal isOpen={true} onClose={jest.fn()} />);
    const refTab = screen.getByRole('button', { name: /endpoints quick reference/i });
    fireEvent.click(refTab);

    expect(screen.getByText('System & Diagnostics')).toBeTruthy();
    expect(screen.getByText('/health')).toBeTruthy();
    expect(screen.getByText('Multimodal Frame Retrieval')).toBeTruthy();
    expect(screen.getByText('/api/v1/search')).toBeTruthy();
    expect(screen.getByText('/api/v1/vqa')).toBeTruthy();
    expect(screen.getByText('/api/v1/trake')).toBeTruthy();
  });

  test('copies cURL command when clicking Copy cURL button', () => {
    Object.assign(navigator, {
      clipboard: {
        writeText: jest.fn().mockImplementation(() => Promise.resolve()),
      },
    });

    render(<ApiDocsModal isOpen={true} onClose={jest.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /endpoints quick reference/i }));

    const copyButtons = screen.getAllByRole('button', { name: /copy curl/i });
    expect(copyButtons.length).toBeGreaterThan(0);
    fireEvent.click(copyButtons[0]);

    expect(navigator.clipboard.writeText).toHaveBeenCalled();
  });

  test('calls onClose when clicking close button', () => {
    const onClose = jest.fn();
    render(<ApiDocsModal isOpen={true} onClose={onClose} />);
    const closeBtn = screen.getByTitle('Close [Esc]');
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
