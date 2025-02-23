import React from 'react';
import { useNavigate } from 'react-router-dom';
import './LoginShenanigans.css'; 

const LoginShenanigans = () => {
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    const username = e.target.elements.username.value;
    const password = e.target.elements.password.value;

    try {
      const response = await fetch('http://127.0.0.1:5000/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      if (response.ok) {
        const data = await response.json();
        alert(data.message);
        navigate('/main');// Displays "Login successful!" if credentials are valid
      } else {
        alert('Invalid credentials.');
      }
    } catch (error) {
      alert('An error occurred.');
    }
  };

  return (
    <div className='wrapper'>
      <form onSubmit={handleSubmit}>
        <h1> TheraBot Login </h1>
        <div className='inputBox'>
          <input type="text" name="username" placeholder='Username' required />
        </div>
        <div className='inputBox'>
          <input type="password" name="password" placeholder='Password' required />
        </div>
        <div className='inputBox'>
          <input type="school" name="school" placeholder='School' optional />
        </div>
        <div className='inputBox'>
          <input type="age" name="age" placeholder='Age' required />
        </div>
        <div className='inputFeeling'>
          <textarea type="feeling" name="feeling" placeholder='Describe how you are feeling in one word'  rows ="2" required />
        </div>
        <div className='remember_forgot'>
          <label>
            <input type='checkbox' /> Remember me
          </label>
          <a href='#'> Forgot Password? </a>
        </div>
        <button type='submit'> Login </button>
        <div className='signupLink'>
          <p> Don't have an account? <a href='#'> Sign Up </a> </p>
        </div>
      </form>
    </div>
  );
};

export default LoginShenanigans;
 