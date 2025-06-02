import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Rocket, BarChart2, Database, Zap, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/contexts/AuthContext';
import { toast } from 'sonner';
import { lazy, Suspense } from 'react';
import LearnMore from '../pages/LearnMore';

// Lazy load animated chart
const AnimatedOrdersChart = lazy(() => import('@/components/AnimatedChart'));

const Home = () => {
  const navigate = useNavigate();
  const { user } = useAuth();

  // Animation variants
  const fadeIn = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { duration: 0.8 } }
  };

  const slideUp = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.6 } }
  };

  const handleNavigation = (path) => {
    if (!user && path !== '/learn-more') {
      toast.error('Please login to access the dashboard');
      navigate('/auth/login');
      return;
    }
    navigate(path);
  };

  return (
    <div className="bg-white min-h-screen relative overflow-x-hidden">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-100">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">
          <Link to="/" className="flex items-center">
            <Rocket className="h-6 w-6 text-blue-600" />
            <span className="ml-2 text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-emerald-500">
              ForecastPro
            </span>
          </Link>
          <nav className="hidden md:flex space-x-8">
            <Link to="/features" className="text-gray-600 hover:text-blue-600 transition-colors">
              Features
            </Link>
            <Link to="/pricing" className="text-gray-600 hover:text-blue-600 transition-colors">
              Pricing
            </Link>
            <Link to="/learn-more" className="text-gray-600 hover:text-blue-600 transition-colors">
              Resources
            </Link>
            <Link to="/contact" className="text-gray-600 hover:text-blue-600 transition-colors">
              Contact
            </Link>
          </nav>
          <div className="flex items-center space-x-4">
            {user ? (
              <Button 
                variant="ghost" 
                onClick={() => navigate('/dashboard')}
                className="text-blue-600 hover:bg-blue-50"
              >
                Dashboard
              </Button>
            ) : (
              <>
                <Button 
                  variant="ghost" 
                  onClick={() => navigate('/auth/login')}
                  className="text-gray-600 hover:bg-gray-50"
                >
                  Sign In
                </Button>
                <Button 
                  onClick={() => navigate('/auth/signup')}
                  className="bg-gradient-to-r from-blue-600 to-emerald-500 hover:from-blue-700 hover:to-emerald-600 text-white"
                >
                  Get Started
                </Button>
              </>
            )}
          </div>
        </div>
      </header>

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
          <motion.div variants={slideUp} className="flex flex-col sm:flex-row justify-center gap-4">
            <Button
              className="bg-gradient-to-r from-blue-600 to-emerald-500 hover:from-blue-700 hover:to-emerald-600 px-8 py-6 text-lg text-white shadow-lg hover:shadow-xl transition-all"
              onClick={() => handleNavigation('/dashboard')}
            >
              <Rocket className="mr-2" />
              Launch Dashboard
              <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              className="border-amber-400 text-amber-600 hover:bg-amber-50 px-8 py-6 text-lg hover:border-amber-500 transition-all"
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
  );
};

export default Home;