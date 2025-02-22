import React from 'react';
import './LoginShenanigans.css'; 
const LoginShenanigans = () => {
  return (
    <div className = 'wrapper'>
        <h1> TheraBot Login </h1>
        <div className = 'inputBox'>
          <input type = "text" placeholder = 'Username' required />
        </div>
        <div className = 'inputBox'>
          <input type = "password" placeholder = 'Password' required />
        </div>
        <div className = 'remember_forgot'>
          <label><input type = 'checkbox' /> Remember me </label>
          <a href = '#'> Forgot Password? </a>
        </div>
        <button type = 'submit'> Login </button>
        <div className = 'signupLink'>
          <p> Don't have an account?  <a href = '#'>  Sign Up </a> </p>
        </div>
    </div>
  );
}

export default LoginShenanigans;