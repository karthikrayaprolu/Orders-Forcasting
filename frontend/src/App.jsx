import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DataStep from './steps/DataStep';
import Home from './pages/Home';
import ProcessStep from './steps/ProcessStep';
import Results from './steps/Results';
import Login from './pages/Login';
import Signup from './pages/Signup';
import LearnMore from './pages/LearnMore';

function App() {
  const [currentStep, setCurrentStep] = useState(0);
  const [forecastData, setForecastData] = useState(null);

  const handleDataUpload = () => {
    setCurrentStep(1);
  };

  const handleForecastComplete = (data) => {
    setForecastData(data);
    setCurrentStep(2);
  };

  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <Routes>
          {/* Main Pages */}
          <Route path="/" element={<Home />} />
          <Route path="/learn-more" element={<LearnMore />} />
          
          {/* Auth Pages */}
          <Route path="/auth/login" element={<Login />} />
          <Route path="/auth/signup" element={<Signup />} />
          
          {/* Dashboard Workflow */}
          <Route
            path="/dashboard"
            element={
              currentStep === 0 ? (
                <DataStep onComplete={handleDataUpload} />
              ) : currentStep === 1 ? (
                <ProcessStep onComplete={handleForecastComplete} />
              ) : (
                <Results forecasts={forecastData} />
              )
            }
          />

          {/* Additional Pages (you can add more as needed) */}
          <Route path="/features" element={<div>Features Page</div>} />
          <Route path="/pricing" element={<div>Pricing Page</div>} />
          <Route path="/contact" element={<div>Contact Page</div>} />
          <Route path="/demo" element={<div>Demo Page</div>} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;