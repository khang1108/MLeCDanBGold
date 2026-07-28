import React from "react";

// Switcher between Conversation (KIS) and Ad-hoc search views.
const TabNavigation = ({ activeTab, onSelectTab }) => {
  return (
    <nav className="tab-navigation" aria-label="Main Navigation">
      <button
        type="button"
        className={`tab-button ${activeTab === "conversation" ? "active" : ""}`}
        onClick={() => onSelectTab("conversation")}
      >
        Conversation
      </button>
      <button
        type="button"
        className={`tab-button ${activeTab === "ad_hoc" ? "active" : ""}`}
        onClick={() => onSelectTab("ad_hoc")}
      >
        Ad-hoc Search
      </button>
    </nav>
  );
};

export default TabNavigation;
