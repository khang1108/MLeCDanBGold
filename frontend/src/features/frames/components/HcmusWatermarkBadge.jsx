import React from 'react';

/**
 * HCMUS & MLeCDanBGold 2026 intellectual property watermark badge.
 * Displayed in empty / idle state across KIS and AVS workspaces.
 */
const HcmusWatermarkBadge = () => (
  <div className="hcmus-copyright-badge" data-testid="hcmus-copyright-badge">
    <img
      src="/hcmus_logo.png"
      alt="HCMUS - Ho Chi Minh University of Science"
      className="hcmus-copyright-logo"
    />
    <div className="hcmus-copyright-content">
      <h3 className="hcmus-copyright-title">MLeCDanBGold · 2026</h3>
      <p className="hcmus-copyright-owner">
        Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM
      </p>
      <p className="hcmus-copyright-owner-en">
        Ho Chi Minh University of Science (VNU-HCM)
      </p>
      <div className="hcmus-copyright-legal">
        <p className="hcmus-copyright-statement">
          © 2026 Team MLeCDanBGold. All Rights Reserved.
        </p>
        <p className="hcmus-copyright-subnote">
          Proprietary Multimodal Video Retrieval & Reasoning System.
          Unauthorized copying, redistribution, or reverse engineering is strictly prohibited.
        </p>
      </div>
    </div>
  </div>
);

export default HcmusWatermarkBadge;
