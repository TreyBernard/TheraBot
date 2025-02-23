import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LoginShenanigans from './components/login/LoginShenaningans';
import MainPage from './components/login/MainPage';
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LoginShenanigans />} />
        <Route path="/main" element={<MainPage />} />
      </Routes>
    </Router>
  );
}

export default App;