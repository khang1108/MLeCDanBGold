import React from 'react';

const MiniChallengePanel = ({ challenge }) => (
  <section className="minichallenge-panel">
    <div className="minichallenge-heading">
      <h3 className="adhoc-sidebar-title">Mini-challenge</h3>
      <span>DRES</span>
    </div>

    <label className="minichallenge-field">
      <span>Session token</span>
      <input
        type="password"
        className="toolbox-input"
        value={challenge.session}
        onChange={(event) => challenge.setSession(event.target.value)}
        autoComplete="off"
        placeholder="Paste session token"
      />
    </label>
    <button
      type="button"
      className="btn-utility minichallenge-connect"
      onClick={challenge.connect}
      disabled={!challenge.session.trim() || challenge.isLoading}
    >
      {challenge.isLoading ? 'Loading…' : 'Load evaluation'}
    </button>

    {challenge.evaluations.length > 0 && (
      <label className="minichallenge-field">
        <span>Evaluation</span>
        <select
          className="toolbox-input"
          value={challenge.evaluationId}
          onChange={(event) => challenge.selectEvaluation(event.target.value)}
          disabled={challenge.isLoading}
        >
          {challenge.evaluations.map((evaluation) => (
            <option key={evaluation.id} value={evaluation.id}>
              {evaluation.name} · {evaluation.status}
            </option>
          ))}
        </select>
      </label>
    )}

    {challenge.currentTask && (
      <div className="minichallenge-task" role="status">
        <strong>{challenge.currentTask.name}</strong>
        <span>
          {challenge.currentTask.taskGroup} · {challenge.currentTask.taskType}
        </span>
      </div>
    )}

    {challenge.currentTask && (
      <label className="minichallenge-field">
        <span>Answer (optional for media-only tasks)</span>
        <textarea
          className="toolbox-input minichallenge-answer"
          value={challenge.answer}
          onChange={(event) => challenge.setAnswer(event.target.value)}
          maxLength={2000}
          placeholder="Enter the QA answer"
        />
      </label>
    )}

    {challenge.error && (
      <p className="minichallenge-message error" role="alert">
        {challenge.error}
      </p>
    )}
    {challenge.notice && (
      <p className="minichallenge-message success" role="status">
        {challenge.notice}
      </p>
    )}
  </section>
);

export default MiniChallengePanel;
