import React, { useState, useEffect } from 'react';
import { requestJson } from '../../../api/client';

const ManualSubmissionBox = () => {
  const [sessionId, setSessionId] = useState('');
  const [sessionStatus, setSessionStatus] = useState('idle');
  const [evaluations, setEvaluations] = useState([]);
  const [selectedTaskString, setSelectedTaskString] = useState('');
  const [errorMsg, setErrorMsg] = useState('');

  const [mediaItemName, setMediaItemName] = useState('');
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(0);
  const [text, setText] = useState('');

  const [submitStatus, setSubmitStatus] = useState('idle');

  const handleLoadSession = async () => {
    if (!sessionId.trim()) return;
    setSessionStatus('loading');
    setErrorMsg('');
    try {
      const evals = await requestJson('/api/v1/minichallenge/evaluations', {
        headers: { 'X-DRES-Session': sessionId.trim() },
      });
      
      if (!Array.isArray(evals) || evals.length === 0) {
        throw new Error("No active evaluations found.");
      }
      
      setEvaluations(evals);

      // Auto-select the first available task across all evals
      for (const evalData of evals) {
        if (evalData.taskTemplates && evalData.taskTemplates.length > 0) {
          setSelectedTaskString(`${evalData.id}::${evalData.taskTemplates[0].name}`);
          break;
        }
      }
      
      setSessionStatus('success');
    } catch (err) {
      setSessionStatus('error');
      setErrorMsg(err.message || 'Failed to load evaluations');
    }
  };

  const handleSubmit = async () => {
    if (evaluations.length === 0 || !selectedTaskString) return;
    setSubmitStatus('loading');
    setErrorMsg('');
    
    const [evalId, taskName] = selectedTaskString.split("::");

    const payload = {
      answerSets: [
        {
          taskName: taskName,
          answers: [
            {
              mediaItemName: mediaItemName.trim(),
              start: Number(start),
              end: Number(end),
              text: text.trim() || null,
            }
          ]
        }
      ]
    };

    try {
      const dresUrl = `http://if-wan4.selab.edu.vn:20740/api/v2/submit/${evalId}?session=${sessionId.trim()}`;
      
      const response = await fetch(dresUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.description || "Submission failed");
      }
      
      setSubmitStatus('success');
      setTimeout(() => {
        setSubmitStatus('idle');
      }, 3000);
    } catch (err) {
      setSubmitStatus('error');
      setErrorMsg(err.message || 'Submission failed.');
    }
  };

  // Helper to determine if selected task is a QA task
  const isQaTask = () => {
    if (!selectedTaskString) return false;
    const [evalId, taskName] = selectedTaskString.split("::");
    const evalData = evaluations.find(e => e.id === evalId);
    if (!evalData) return false;
    const task = evalData.taskTemplates?.find(t => t.name === taskName);
    if (!task) return false;
    return task.taskGroup?.toLowerCase().includes('qa') || 
           task.taskGroup?.toLowerCase().includes('vqa') || 
           task.taskType?.toLowerCase().includes('qa');
  };

  return (
    <div className="toolbox-section" style={{ padding: '12px', gap: '12px', border: '1px solid var(--color-primary)', backgroundColor: 'var(--color-surface)', marginTop: '8px' }}>
      <div style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--color-primary)', marginBottom: '8px', borderBottom: '1px solid var(--color-hairline)', paddingBottom: '4px' }}>
        Mini-Challenge Submit
      </div>

      {errorMsg && <div style={{ padding: '8px', backgroundColor: '#ffebee', color: '#c62828', borderRadius: '4px', fontSize: '12px' }}>{errorMsg}</div>}
      
      {sessionStatus !== 'success' ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <input 
            type="text" 
            value={sessionId} 
            onChange={e => setSessionId(e.target.value)} 
            placeholder="DRES session ID"
            style={{ padding: '6px 8px', borderRadius: '4px', border: '1px solid var(--color-hairline)', width: '100%', fontSize: '12px' }}
          />
          <button 
            onClick={handleLoadSession}
            disabled={!sessionId.trim() || sessionStatus === 'loading'}
            className="btn-primary"
            style={{ padding: '6px 12px', fontSize: '12px', width: '100%' }}
          >
            {sessionStatus === 'loading' ? 'Loading...' : 'Load Evaluations'}
          </button>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          
          {evaluations.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--color-ink)' }}>Select Task:</label>
              <select 
                value={selectedTaskString} 
                onChange={e => setSelectedTaskString(e.target.value)}
                style={{ padding: '6px 8px', borderRadius: '4px', border: '1px solid var(--color-hairline)', fontSize: '12px', width: '100%' }}
              >
                {evaluations.map(e => (
                  <optgroup key={e.id} label={`Eval: ${e.name}`}>
                    {e.taskTemplates?.map(t => (
                      <option key={`${e.id}::${t.name}`} value={`${e.id}::${t.name}`}>
                        {t.name} ({t.taskGroup})
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>
          )}

          <input 
            type="text" 
            value={mediaItemName} 
            onChange={e => setMediaItemName(e.target.value)} 
            placeholder="mediaItemName (e.g. L30_V095)"
            style={{ padding: '6px 8px', borderRadius: '4px', border: '1px solid var(--color-hairline)', fontSize: '12px' }}
          />

          <div style={{ display: 'flex', gap: '8px' }}>
            <input 
              type="number" 
              min="0"
              value={start} 
              onChange={e => setStart(e.target.value)} 
              placeholder="Start ms"
              title="Start ms"
              style={{ padding: '6px 8px', borderRadius: '4px', border: '1px solid var(--color-hairline)', fontSize: '12px', flex: 1, minWidth: 0 }}
            />
            <input 
              type="number" 
              min="0"
              value={end} 
              onChange={e => setEnd(e.target.value)} 
              placeholder="End ms"
              title="End ms"
              style={{ padding: '6px 8px', borderRadius: '4px', border: '1px solid var(--color-hairline)', fontSize: '12px', flex: 1, minWidth: 0 }}
            />
          </div>

          {isQaTask() && (
            <textarea
              value={text} 
              onChange={e => setText(e.target.value)} 
              placeholder="Answer text"
              rows={2}
              style={{ padding: '6px 8px', borderRadius: '4px', border: '1px solid var(--color-hairline)', fontSize: '12px', resize: 'vertical' }}
            />
          )}

          <button 
            onClick={handleSubmit}
            disabled={submitStatus === 'loading' || !mediaItemName.trim()}
            className="btn-primary"
            style={{ padding: '8px', fontSize: '12px', backgroundColor: submitStatus === 'success' ? '#2e7d32' : undefined }}
          >
            {submitStatus === 'loading' ? 'Submitting...' : submitStatus === 'success' ? '✓ Success' : 'Submit'}
          </button>
        </div>
      )}
    </div>
  );
};

export default ManualSubmissionBox;
