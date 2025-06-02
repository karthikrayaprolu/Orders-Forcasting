import React, { createContext, useContext, useEffect, useState } from 'react';
import { onAuthStateChanged } from 'firebase/auth';
import { auth } from '../config/firebase';
import { signUp, login, loginWithGoogle, logout } from '../services/auth';

const AuthContext = createContext({});

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (user) => {
            setUser(user);
            setLoading(false);
        });

        return unsubscribe;
    }, []);

    const handleSignUp = async (email, password) => {
        try {
            setError(null);
            await signUp(email, password);
        } catch (error) {
            setError(error.message);
            throw error;
        }
    };

    const handleLogin = async (email, password) => {
        try {
            setError(null);
            await login(email, password);
        } catch (error) {
            setError(error.message);
            throw error;
        }
    };

    const handleGoogleLogin = async () => {
        try {
            setError(null);
            await loginWithGoogle();
        } catch (error) {
            setError(error.message);
            throw error;
        }
    };

    const handleLogout = async () => {
        try {
            setError(null);
            await logout();
        } catch (error) {
            setError(error.message);
            throw error;
        }
    };

    const value = {
        user,
        loading,
        error,
        signUp: handleSignUp,
        login: handleLogin,
        loginWithGoogle: handleGoogleLogin,
        logout: handleLogout
    };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
};