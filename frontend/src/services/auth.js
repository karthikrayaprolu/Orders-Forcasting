import { 
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signInWithPopup,
    GoogleAuthProvider,
    signOut
} from 'firebase/auth';
import { auth } from '../config/firebase';

const googleProvider = new GoogleAuthProvider();

export const signUp = async (email, password) => {
    try {
        console.log('Attempting signup with:', { email, hasPassword: !!password });
        const userCredential = await createUserWithEmailAndPassword(auth, email, password);
        console.log('Signup successful:', userCredential.user.uid);
        return userCredential.user;
    } catch (error) {
        console.error('Signup error:', error.code, error.message);
        switch (error.code) {
            case 'auth/email-already-in-use':
                throw new Error('This email is already registered. Please try logging in.');
            case 'auth/invalid-email':
                throw new Error('Invalid email address format.');
            case 'auth/weak-password':
                throw new Error('Password should be at least 6 characters long.');
            default:
                throw new Error(error.message);
        }
    }
};

export const login = async (email, password) => {
    try {
        console.log('Attempting login with:', { email, hasPassword: !!password });
        const userCredential = await signInWithEmailAndPassword(auth, email, password);
        console.log('Login successful:', userCredential.user.uid);
        return userCredential.user;
    } catch (error) {
        console.error('Login error:', error.code, error.message);
        switch (error.code) {
            case 'auth/user-not-found':
                throw new Error('No account found with this email.');
            case 'auth/wrong-password':
                throw new Error('Incorrect password.');
            case 'auth/invalid-email':
                throw new Error('Invalid email address format.');
            case 'auth/too-many-requests':
                throw new Error('Too many failed attempts. Please try again later.');
            default:
                throw new Error('Failed to sign in. Please check your credentials.');
        }
    }
};

export const loginWithGoogle = async () => {
    try {
        console.log('Attempting Google login');
        const result = await signInWithPopup(auth, googleProvider);
        console.log('Google login successful:', result.user.uid);
        return result.user;
    } catch (error) {
        console.error('Google login error:', error.code, error.message);
        switch (error.code) {
            case 'auth/popup-closed-by-user':
                throw new Error('Sign-in popup was closed before completing.');
            case 'auth/popup-blocked':
                throw new Error('Sign-in popup was blocked by the browser.');
            case 'auth/cancelled-popup-request':
                throw new Error('Another sign-in popup is already open.');
            case 'auth/account-exists-with-different-credential':
                throw new Error('An account already exists with the same email but different sign-in credentials.');
            default:
                throw new Error('Failed to sign in with Google.');
        }
    }
};

export const logout = async () => {
    try {
        await signOut(auth);
    } catch (error) {
        throw new Error(error.message);
    }
};