import React from 'react';

const MiniChallengePanel = ({ challenge }) => {
  const isConnected = Boolean(challenge.session && challenge.evaluations.length > 0);

  return (
    <section className="minichallenge-panel">
      <div className="minichallenge-heading">
        <h3 className="adhoc-sidebar-title">Mini-challenge</h3>
        <span className="minichallenge-badge">DRES</span>
      </div>

      {!isConnected ? (
        <div className="minichallenge-form">
          <label className="minichallenge-field">
            <span>Session ID (tự động lấy từ .env hoặc nhập trực tiếp)</span>
            <input
              type="text"
              className="toolbox-input"
              value={challenge.session}
              onChange={(e) => challenge.setSession(e.target.value)}
              placeholder="Nhập hoặc dán Session ID..."
            />
          </label>
          <button
            type="button"
            className="btn-utility minichallenge-connect"
            onClick={challenge.connect}
            disabled={!challenge.session.trim() || challenge.isLoading}
          >
            {challenge.isLoading ? 'Đang tải evaluation…' : 'Tải Evaluation Run'}
          </button>
        </div>
      ) : (
        <div className="minichallenge-connected-box">
          <div className="minichallenge-session-header">
            <span className="minichallenge-status-dot online" />
            <span className="minichallenge-user-info">
              {challenge.username ? `User: ${challenge.username}` : 'DRES Session Active'}
            </span>
            <button
              type="button"
              className="minichallenge-logout-btn"
              onClick={challenge.connect}
              disabled={challenge.isLoading}
              title="Tải lại evaluations"
            >
              Làm mới
            </button>
          </div>

          <label className="minichallenge-field">
            <span>Evaluation Run</span>
            <select
              className="toolbox-input"
              value={challenge.evaluationId}
              onChange={(event) => challenge.selectEvaluation(event.target.value)}
              disabled={challenge.isLoading}
            >
              {challenge.evaluations.map((evaluation) => (
                <option key={evaluation.id} value={evaluation.id}>
                  {evaluation.name} ({evaluation.status})
                </option>
              ))}
            </select>
          </label>

          {challenge.currentTask ? (
            <div className="minichallenge-task" role="status">
              <div className="minichallenge-task-header">
                <strong>{challenge.currentTask.name}</strong>
                <button
                  type="button"
                  className="minichallenge-refresh-btn"
                  onClick={challenge.refreshTask}
                  disabled={challenge.isLoading}
                  title="Cập nhật task hiện tại"
                >
                  Làm mới task
                </button>
              </div>
              <span>
                Group: {challenge.currentTask.taskGroup} · Type: {challenge.currentTask.taskType}
                {challenge.currentTask.duration ? ` · ${challenge.currentTask.duration}s` : ''}
              </span>
            </div>
          ) : (
            <div className="minichallenge-task-loading">
              <span>Chưa có task nào đang mở.</span>
              <button
                type="button"
                className="minichallenge-refresh-btn"
                onClick={challenge.refreshTask}
                disabled={challenge.isLoading}
              >
                Kiểm tra task
              </button>
            </div>
          )}

          <label className="minichallenge-field">
            <span>Đáp án văn bản (cho VQA / Text QA)</span>
            <textarea
              className="toolbox-input minichallenge-answer"
              value={challenge.answer}
              onChange={(event) => challenge.setAnswer(event.target.value)}
              maxLength={2000}
              placeholder="Nhập câu trả lời hoặc để trống nếu chỉ nộp video/frame"
            />
          </label>
        </div>
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
};

export default MiniChallengePanel;


