import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import DataStep from './steps/DataStep';
import ProcessStep from './steps/ProcessStep';
import Results from './steps/Results';

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
          <Route path="/" element={
            currentStep === 0 ? (
              <DataStep onComplete={handleDataUpload} />
            ) : currentStep === 1 ? (
              <ProcessStep onComplete={handleForecastComplete} />
            ) : (
              <Results forecasts={forecastData} />
            )
          } />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
