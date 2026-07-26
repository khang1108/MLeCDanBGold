import React from 'react';
import ToolBox from './ToolBox';

// Owns the modal shell while ToolBox owns the actual controls.
const OptionsDrawer = ({ isOpen, onClose, ...toolBoxProps }) => {
  if (!isOpen) return null;
  return <div className="options-scrim" onClick={onClose}><aside className="options-drawer" onClick={(event) => event.stopPropagation()}><div className="options-drawer-header"><div><p className="workspace-eyebrow">Search setup</p><h2>Options</h2></div><button className="btn-utility" onClick={onClose}>Close</button></div><ToolBox {...toolBoxProps} /></aside></div>;
};

export default OptionsDrawer;