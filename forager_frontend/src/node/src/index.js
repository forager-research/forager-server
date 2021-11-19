import React from 'react';
import ReactDOM from 'react-dom';
import DatasetList from './DatasetList';
import App from './App';
import { UserProvider } from './UserContext';

import {
  BrowserRouter as Router,
  Switch,
  Route,
  Redirect,
} from "react-router-dom";

import TimeAgo from "javascript-time-ago";
import en from "javascript-time-ago/locale/en";
TimeAgo.addDefaultLocale(en);

ReactDOM.render(
  <React.StrictMode>
    <UserProvider>
      <Router>
        <Switch>
          <Route exact path="/" children={<DatasetList />} />
          <Route path="/:datasetName" children={<App />} />
        </Switch>
      </Router>
    </UserProvider>
  </React.StrictMode>,
  document.getElementById('root')
);
