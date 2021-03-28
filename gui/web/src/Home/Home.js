import React, { Fragment, useState } from 'react';
import { Row, Col } from 'antd';
  
import Layout from '../Layout';
import Welcome from './Welcome';
import PredictionsBarChart from './PredictionsBarChart';
import TrafficMap from './TrafficMap';
import SearchPanel from './SearchPanel';
import PredictionSummary from './PredictionSummary';


export default function ({ predictions, sites, mapData, loadPredictions }) {
    const [source, setSource] = useState(null);
    const [dest, setDest] = useState(null);
    const [dataTransferred, setDataTransferred] = useState(null);
    const [selection, setSelection] = useState(null);
    const [highlight, setHighlight] = useState(null);
    const [path, setPath] = useState(null);

    function handleSourceChange(selected) {
      setSource(selected);
    }

    function handleDestChange(selected) {
      setDest(selected);
    }

    function handleDataTransferredChange(selected) {
      setDataTransferred(selected);
    }

    function handleBarSelection(selection) {
      if (selection === null) {
        setSelection(null);
        setPath(null);
        
        return;
      }

      const selectedPath = selection.event.get('path');
      setSelection(selection);
      setPath(selectedPath);
    }

    return (
      <Layout>
        <Row gutter={[25, 25]}>
          <Col className="gutter-row" span={4}>
            <div className="page-section" style={{ height: 320 }}>
              <SearchPanel
                sites={sites}
                source={source}
                handleSourceChange={handleSourceChange}
                dest={dest}
                handleDestChange={handleDestChange}
                dataTransferred={dataTransferred}
                handleDataTransferredChange={handleDataTransferredChange}
                loadPredictions={loadPredictions}
              />
            </div>
          </Col>
          <Col className="gutter-row" span={20}>
            <div
              className="page-section"
              style={{ height: 320, position: 'relative' }}
            >
              {
                predictions === null ?
                <Welcome />
                :
                (<Fragment>
                  <Row>
                    <Col className="gutter-row" span={17} style={{ padding: 20 }}>
                      <div style={{ padding: '50px 10px' }}>
                        <PredictionsBarChart
                          predictions={predictions}
                          selection={selection}
                          handleBarSelection={handleBarSelection}
                          highlight={highlight}
                          handleBarHighlight={setHighlight}
                          setPath={setPath}
                        />
                      </div>
                    </Col>
                    <Col className="gutter-row" span={7}>
                    <div className="vertical-rule" style={{ height: 320 }}>
                        <PredictionSummary
                        openTrendModal={() => {}}
                        source={source.label}
                        dest={dest.label}
                        dataTransfer={dataTransferred.label}
                        bestTime="9PM"
                        />
                    </div>
                    </Col>
                  </Row>
                </Fragment>)
              }
            </div>
          </Col>
          <Col span={24}>
            <div className="page-section">
              <TrafficMap mapData={mapData} path={path} />
            </div>
          </Col>
        </Row>
      </Layout>
    );
  }
