import React from "react";

const scoreLabels = [
  ["visual", "Visual"],
  ["caption", "Caption"],
  ["ocr", "OCR"],
  ["asr", "ASR"],
  ["fusion", "Fusion"],
  ["reranker", "Rerank"],
];

// Reused by the card tooltip and selected-frame inspector.
const ScoreBreakdown = ({
  scores,
  className = "score-tooltip-table",
  asRows = false,
}) => {
  const rows = scoreLabels.filter(
    ([key]) => scores[key] !== null && scores[key] !== undefined,
  );
  if (asRows)
    return rows.map(([key, label]) => (
      <div key={key} className="inspector-score-row">
        <span className="score-row-name">{label}</span>
        <span className="score-row-val">{scores[key].toFixed(2)}</span>
      </div>
    ));
  return (
    <table className={className}>
      <tbody>
        {rows.map(([key, label]) => (
          <tr key={key}>
            <td className="score-name">{label}:</td>
            <td className="score-value">{scores[key].toFixed(2)}</td>
          </tr>
        ))}
        <tr className="score-tooltip-divider">
          <td colSpan="2" />
        </tr>
        <tr className="score-tooltip-highlight">
          <td className="score-name">Final:</td>
          <td className="score-value">{scores.final.toFixed(2)}</td>
        </tr>
      </tbody>
    </table>
  );
};

export default ScoreBreakdown;
