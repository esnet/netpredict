import React, { Fragment } from 'react';
import { Layout, Menu} from 'antd';

import logo from './images/netpredict.png';

const { Header, Footer, Sider, Content } = Layout;

export default function ({ children }) {
  return (
    <Layout className="main-layout">
      <Header className="main-header">
        <div className="header-logo-wrapper">
          <img src={logo} className="header-logo" />
          <h1 className="header-title">NetPredict</h1>
        </div>
        <Menu theme="light" mode="horizontal" className="header-menu">
          <Menu.Item key="1" className="header-menu-item">
            Home
          </Menu.Item>
          <Menu.Item key="2" className="header-menu-item">
            Real-time Error Tracking
          </Menu.Item>
           <Menu.Item key="3" className="header-menu-item">
            About Us
          </Menu.Item>
          <Menu.SubMenu title="View Trends" className="header-menu-item">
            <Menu.Item key="4" className="header-Submenu-item ">
            Coming Soon
           </Menu.Item>
          </Menu.SubMenu>
          <Menu.Item key="5" className="header-menu-item">
            Sites
          </Menu.Item>
        </Menu>
      </Header>
      <Content className="main-content">{children}</Content>
      <Footer className="page-footer">© Copyright NetPredict 2020</Footer>
    </Layout>
  );
}
