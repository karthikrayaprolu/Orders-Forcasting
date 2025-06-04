import React, { lazy, Suspense, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Rocket, BarChart2, Database, Zap, ChevronRight, LogOut, User } from 'lucide-react';
import { Button } from '../components/ui/button';
import { useAuth } from '../contexts/AuthContext';
import { toast } from 'sonner';
import LearnMore from '../pages/LearnMore';
import Dashboard from '../pages/Dashboard';
import NotFound from '../pages/NotFound';

// Lazy load animated chart
const AnimatedOrdersChart = lazy(() => import('../components/AnimatedChart'));

const Home = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [showDashboard, setShowDashboard] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [forecastData, setForecastData] = useState(null);

  const handleSignOut = async () => {
    try {
      await logout();
      setShowDashboard(false);
      navigate('/');
      toast.success('Successfully signed out');
    } catch (error) {
      console.error('Logout error:', error);
      toast.error('Failed to sign out');
    }
  };

  // Animation variants
  const fadeIn = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 0.8 } }
  };

  const slideUp = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.6 } }
  };
  const [showNotFound, setShowNotFound] = useState(false);
  const [notFoundPath, setNotFoundPath] = useState('');

  const handleNavigation = (path) => {
    if (path === '/dashboard') {
      if (!user) {
        toast.error('Please login to access the dashboard');
        navigate('/login');
      } else {
        setShowDashboard(true);
        setShowNotFound(false);
      }    } else if (['/features', '/pricing', '/resources', '/contact'].includes(path)) {
      setShowNotFound(true);
      setNotFoundPath(path.substring(1));
      setShowDashboard(false);
    } else if (path === '/') {
      setShowNotFound(false);
      setShowDashboard(false);
      navigate('/');
    } else {
      setShowNotFound(false);
      navigate(path);
    }
  };

  const handleStepComplete = (data) => {
    if (data) {
      setForecastData(data);
    }
    setCurrentStep(prev => prev + 1);
  };

  return (
    <div className="bg-white h-screen flex flex-col overflow-hidden">
      {/* Header - Fixed */}
      <header className="flex-none bg-white border-b border-gray-100 z-50">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">          <button onClick={() => handleNavigation('/')} className="flex items-center">
            <Rocket className="h-6 w-6 text-blue-600" />
            <span className="ml-2 text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-emerald-500">
              ForecastPro
            </span>
          </button>
          <nav className="hidden md:flex space-x-8">
            {user && (
              <button 
                onClick={() => setShowDashboard(true)} 
                className={`text-gray-600 hover:text-blue-600 transition-colors ${showDashboard ? 'text-blue-600 font-medium' : ''}`}
              >
                Dashboard
              </button>
            )}            <button onClick={() => handleNavigation('/features')} className="text-gray-600 hover:text-blue-600 transition-colors">
              Features
            </button>
            <button onClick={() => handleNavigation('/pricing')} className="text-gray-600 hover:text-blue-600 transition-colors">
              Pricing
            </button>
            <button onClick={() => handleNavigation('/resources')} className="text-gray-600 hover:text-blue-600 transition-colors">
              Resources
            </button>
            <button onClick={() => handleNavigation('/contact')} className="text-gray-600 hover:text-blue-600 transition-colors">
              Contact
            </button>
          </nav>
          <div className="flex items-center space-x-4">            {user ? (
              <div className="relative group">
                <button className="p-1 rounded-full hover:bg-gray-100 transition-colors">
                  <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center text-sm font-medium">
                    {user.email?.[0].toUpperCase()}
                  </div>
                </button>                <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl border border-gray-100 invisible group-hover:visible transition-all z-50">
                  <div className="px-4 py-2 text-sm text-gray-700 border-b border-gray-100 truncate bg-gray-50">
                    {user?.email}
                  </div>
                  <button 
                    type="button"
                    onClick={handleSignOut}
                    className="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
                  >
                    <LogOut className="h-4 w-4" />
                    Sign Out
                  </button>
                </div>
              </div>
            ) : (
              <>
                <Button 
                  variant="ghost" 
                  onClick={() => navigate('/login')}
                  className="text-gray-600 hover:bg-gray-50"
                >
                  Sign In
                </Button>
                <Button 
                  onClick={() => navigate('/signup')}
                  className="bg-gradient-to-r from-blue-600 to-emerald-500 hover:from-blue-700 hover:to-emerald-600 text-white"
                >
                  Get Started
                </Button>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Scrollable Content Area */}      <div className="flex-1 overflow-y-auto">
        {showDashboard ? (
          <Dashboard
            currentStep={currentStep}
            setCurrentStep={setCurrentStep}
            forecastData={forecastData}
            onStepComplete={handleStepComplete}
          />
        ) : showNotFound ? (
          <div className="container mx-auto px-6 py-12">
            <NotFound path={notFoundPath} />
          </div>
        ) : (
          <div className="min-h-full">
            {/* Animated background elements */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
              <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-50 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob"></div>
              <div className="absolute top-1/3 right-1/4 w-64 h-64 bg-amber-50 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-2000"></div>
              <div className="absolute bottom-1/4 left-1/2 w-64 h-64 bg-emerald-50 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-4000"></div>
            </div>

            {/* Hero Section */}
            <section className="relative container mx-auto px-6 py-24 text-center z-10">
              <motion.div
                initial="hidden"
                animate="visible"
                variants={fadeIn}
                className="max-w-4xl mx-auto"
              >
                <h1 className="text-4xl md:text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-800 to-emerald-600">
                  Advanced Orders Forecasting
                </h1>
                <p className="text-lg md:text-xl text-gray-600 mb-10">
                  Harness AI-powered predictions to optimize your inventory and maximize revenue
                </p>
                <motion.div variants={slideUp} className="flex flex-col sm:flex-row justify-center gap-4">                  <Button
                    className="bg-gradient-to-r from-blue-600 to-emerald-500 hover:from-blue-700 hover:to-emerald-600 text-white shadow-lg hover:shadow-xl transition-all"
                    onClick={() => handleNavigation('/dashboard')}
                  >
                    <Rocket className="mr-2" />
                    Launch Dashboard
                    <ChevronRight className="ml-2 h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    className="border-amber-400 text-amber-600 hover:bg-amber-50 hover:border-amber-500 transition-all"
                    onClick={() => handleNavigation('/learn-more')}
                  >
                    <BarChart2 className="mr-2" />
                    How It Works
                  </Button>
                </motion.div>
              </motion.div>

              {/* Animated chart preview */}
              <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                className="mt-16 mx-auto max-w-4xl"
              >
                <Suspense fallback={<div className="text-gray-500 text-sm">Loading forecast visualization...</div>}>
                  <AnimatedOrdersChart />
                </Suspense>
              </motion.div>
            </section>

            {/* Features Section */}
            <section className="bg-gradient-to-b from-white to-blue-50 py-24 z-10 relative">
              <div className="container mx-auto px-6">
                <motion.h2
                  initial="hidden"
                  whileInView="visible"
                  variants={slideUp}
                  viewport={{ once: true }}
                  className="text-3xl font-bold text-center mb-16 text-blue-900"
                >
                  Powerful Forecasting Features
                </motion.h2>
                <div className="grid md:grid-cols-3 gap-8">
                  {[
                    {
                      icon: <Database size={32} className="text-emerald-500" />,
                      title: "Historical Data Analysis",
                      description: "Upload years of order history to train accurate models",
                      color: "from-blue-100 to-blue-50",
                    },
                    {
                      icon: <Zap size={32} className="text-amber-500" />,
                      title: "Real-time Predictions",
                      description: "Get daily updated forecasts based on latest trends",
                      color: "from-emerald-100 to-emerald-50",
                    },
                    {
                      icon: <BarChart2 size={32} className="text-blue-500" />,
                      title: "Visual Analytics",
                      description: "Interactive charts and exportable reports",
                      color: "from-amber-100 to-amber-50",
                    },
                  ].map((feature, index) => (
                    <motion.div
                      key={index}
                      initial="hidden"
                      whileInView="visible"
                      variants={slideUp}
                      transition={{ delay: index * 0.1 }}
                      viewport={{ once: true }}
                      className={`bg-gradient-to-br ${feature.color} p-8 rounded-xl shadow-sm hover:shadow-md transition-all border border-gray-100`}
                    >
                      <div className="flex justify-center mb-6">
                        <div className="p-4 bg-white rounded-full shadow-sm">
                          {feature.icon}
                        </div>
                      </div>
                      <h3 className="text-xl font-semibold mb-3 text-center text-blue-800">
                        {feature.title}
                      </h3>
                      <p className="text-gray-600 text-center">{feature.description}</p>
                    </motion.div>
                  ))}
                </div>
              </div>
            </section>

            {/* CTA Section */}
            <section className="bg-gradient-to-b from-blue-50 to-white py-24 relative z-10">
              <div className="container mx-auto px-6 text-center">
                <motion.h2
                  initial="hidden"
                  whileInView="visible"
                  variants={slideUp}
                  viewport={{ once: true }}
                  className="text-3xl font-bold mb-6 text-blue-900"
                >
                  Ready to Transform Your Business?
                </motion.h2>
                <motion.p
                  initial="hidden"
                  whileInView="visible"
                  variants={slideUp}
                  transition={{ delay: 0.1 }}
                  viewport={{ once: true }}
                  className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto"
                >
                  Join industry leaders using our platform to predict demand with 95% accuracy
                </motion.p>
                <motion.div
                  initial="hidden"
                  whileInView="visible"
                  variants={slideUp}
                  transition={{ delay: 0.2 }}
                  viewport={{ once: true }}
                  className="flex flex-col sm:flex-row justify-center gap-4"
                >
                  <Button
                    className="bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white px-10 py-6 text-lg font-bold shadow-lg hover:shadow-xl transition-all"
                    onClick={() => navigate('/auth/signup')}
                  >
                    Start Your Free Trial
                  </Button>
                  <Button
                    variant="outline"
                    className="border-blue-400 text-blue-600 hover:bg-blue-50 px-10 py-6 text-lg hover:border-blue-500 transition-all"
                    onClick={() => navigate('/demo')}
                  >
                    Request Live Demo
                  </Button>
                </motion.div>
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  );
};

export default Home;