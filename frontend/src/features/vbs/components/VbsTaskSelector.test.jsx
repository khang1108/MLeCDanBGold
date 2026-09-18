import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import VbsTaskSelector from './VbsTaskSelector';

const evaluations = [{
  id: 'eval-1',
  name: 'VBS',
  taskTemplates: [
    { name: 'KIS task', taskGroup: 'KIS', taskType: 'KIS', duration: 300 },
    { name: 'AVS task', taskGroup: 'AVS', taskType: 'AVS', duration: 300 },
  ],
}];

describe('VbsTaskSelector', () => {
  test('returns the complete selected task object through onRequestChange', () => {
    const onRequestChange = jest.fn();
    render(
      <VbsTaskSelector
        connectedUserId="team-a"
        evaluations={evaluations}
        selectedTask={{
          evaluationId: 'eval-1',
          evaluationName: 'VBS',
          taskName: 'KIS task',
          taskGroup: 'KIS',
          taskType: 'KIS',
          duration: 300,
        }}
        onRequestChange={onRequestChange}
      />,
    );

    fireEvent.change(screen.getByLabelText('Select evaluation task'), {
      target: { value: 'eval-1:AVS task' },
    });

    expect(onRequestChange).toHaveBeenCalledWith({
      evaluationId: 'eval-1',
      evaluationName: 'VBS',
      taskName: 'AVS task',
      taskGroup: 'AVS',
      taskType: 'AVS',
      duration: 300,
    });
  });
});
