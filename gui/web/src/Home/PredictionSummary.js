import React from 'react';
import { Button } from 'antd';

export default function({ openTrendModal, source, dest, dataTransferred, bestTime }) {
    return (
        <div className="prediction-summary">
            <p className="prediction-summary-title">Prediction:</p>
            
            <p className="prediction-summary-main-text">The best time to schedule a transfer
            from <b>{source}</b> to <b>{dest}</b> of size <b>{dataTransferred}</b>
            is by <b>{bestTime}</b>.</p>
            
            <p className="prediction-summary-subtext">Showing predicted best paths from source to destination. For Example 1 Tb transfer would take about 6.2 minutes. To view the trend, please click on the button below.</p>

            <Button className="prediction-summary-button" type="primary" onClick={openTrendModal}>View trend</Button>
        </div>
    );
}