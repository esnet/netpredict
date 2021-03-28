import React from 'react';
import Select from 'react-select';
import { Button } from 'antd';

export default function ({
    source,
    handleSourceChange,
    dest,
    handleDestChange,
    dataTransferred,
    handleDataTransferredChange,
    loadPredictions,
    sites
}) {
    const options = sites.map((site) => {
        return {
          value: site,
          label: site,
        };
      });
  
      const dataTransferredOptions = [
        { value: { from: 10, to: 20}, label: '1GB -  10GB'},
        { value: { from: 10, to: 20}, label: '10GB - 20GB'},
        { value: { from: 20, to: 40}, label: '20GB - 40GB'},
        { value: { from: 40, to: 60}, label: '40GB - 60GB'},
        { value: { from: 40, to: 60}, label: '60GB - 80GB'},
        { value: { from: 40, to: 60}, label: '80GB - 100GB'},
        { value: { from: 40, to: 60}, label: '100GB - 120GB'},
      ];

      const searchDisabled = source === null || dest === null || dataTransferred === null;

    return (
        <form className="search-form">
          <b>Select search criteria</b>
          <hr />
          <div className="form-group">
            <label htmlFor="source">Source</label>
            <Select
              className="search-select"
              id="source"
              placeholder=""
              options={options}
              value={source}
              onChange={handleSourceChange}
            />
          </div>
          <div className="form-group">
            <label htmlFor="dest">Destination</label>
            <Select
              id="dest"
              placeholder=""
              options={options}
              value={dest}
              onChange={handleDestChange}
            />
          </div>
          <div className="form-group">
            <label htmlFor="data-transferred">Data Transferred</label>
            <Select
              id="data-transferred"
              placeholder=""
              options={dataTransferredOptions}
              value={dataTransferred}
              onChange={handleDataTransferredChange}
            />
          </div>
          <div className="form-submit">
            <Button type="primary" onClick={loadPredictions} disabled={searchDisabled}>Search</Button>
          </div>
        </form>
      );
}