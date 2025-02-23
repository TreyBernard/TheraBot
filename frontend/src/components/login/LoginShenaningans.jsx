import React from "react";
import { useNavigate } from "react-router-dom";
import "./LoginShenanigans.css";

const LoginShenanigans = () => {
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        const username = e.target.elements.username.value;
        const password = e.target.elements.password.value;
        const school = e.target.elements.school.value;
        const age = e.target.elements.age.value;
        const feeling = e.target.elements.feeling.value;

        try {
            const response = await fetch("http://127.0.0.1:5000/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, password }),
            });

            if (response.ok) {
                const data = await response.json();
                alert(data.message);

                // Navigate to Main Page with user data
                navigate("/main", {
                    state: {
                        username,
                        school,
                        age,
                        feeling
                    }
                });

            } else {
                alert("Invalid credentials.");
            }
        } catch (error) {
            alert("An error occurred.");
        }
    };

    return (
        <div className="wrapper">
            <form onSubmit={handleSubmit}>
                <h1>TheraBot Login</h1>
                <div className="inputBox">
                    <input type="text" name="username" placeholder="Username" required />
                </div>
                <div className="inputBox">
                    <input type="password" name="password" placeholder="Password" required />
                </div>
                <div className="inputBox">
                    <input type="text" name="school" placeholder="School" />
                </div>
                <div className="inputBox">
                    <input type="number" name="age" placeholder="Age" required />
                </div>
                <div className="inputFeeling">
                    <textarea name="feeling" placeholder="Describe how you are feeling in one word" rows="2" required />
                </div>
                <button type="submit">Login</button>
            </form>
        </div>
    );
};

export default LoginShenanigans;
