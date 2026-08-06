import React from "react";

// Switcher between the two competition workflows.
const TabNavigation = ({ activeTab, onSelectTab }) => {
  return (
    <nav className="tab-navigation" aria-label="Main Navigation">
      <button
        type="button"
        className={`tab-button ${activeTab === "kis" ? "active" : ""}`}
        onClick={() => onSelectTab("kis")}
      >
        KIS
      </button>
      <button
        type="button"
        className={`tab-button ${
          activeTab === "vqa" ? "active" : ""
        }`}
        onClick={() => onSelectTab("vqa")}
      >
        VQA
      </button>
    </nav>
  );
};

export default TabNavigation;
