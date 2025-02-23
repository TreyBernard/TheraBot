<<<<<<< HEAD
import React, { useState, useEffect } from "react";
import axios from "axios";
import "./mainpage.css"; 
import avatar1 from "./cat1.png";
import avatar2 from "./cat2.png";
=======
import React, { useState, useEffect } from 'react';
import './mainpage.css'; 
import avatar1 from './cat1.png';
import avatar2 from './cat2.png';
import axios from "axios";
>>>>>>> 649e19b4 (functioning chatbot shenanigans)

const MainPage = () => {
    const [avatar, setAvatar] = useState(avatar1);
    const [botResponse, setBotResponse] = useState("Hello, how are you today?");
    const [userMessage, setUserMessage] = useState(""); // User input
    const [loading, setLoading] = useState(false); // Loading state

<<<<<<< HEAD
    // Avatar animation (blinking effect)
    useEffect(() => {
        const interval = setInterval(() => {
            setAvatar((prev) => (prev === avatar1 ? avatar2 : avatar1));
        }, 200);
        return () => clearInterval(interval);
    }, []);
=======
  useEffect(() => {
    const interval = setInterval(() => {
        setAvatar((prev) => prev === avatar1 ? avatar2 : avatar1);
}, 150);
    return () => clearInterval(interval);
},[]);
    const [userMessage, setUserMessage] = useState(""); // User input
    const [loading, setLoading] = useState(false); // Loading state

    // Function to send user input to Flask API
    const sendMessage = async () => {
        if (userMessage.trim() === "") return; // Prevent empty messages
        setLoading(true); // Show loading state
        setBotResponse("Thinking..."); // Temporary response

        try {
            const response = await axios.post("http://localhost:5000/generate", {
                user_info: { name: "John", age: 25, school: "GT", feeling: "anxious" },
                emotion: "stressed",
                message: userMessage,
            });

            setBotResponse(response.data.response); // Update with AI response
        } catch (error) {
            console.error("Error fetching response:", error);
            setBotResponse("Oops! Something went wrong.");
        } finally {
            setLoading(false); // Reset loading state
        }
    };
>>>>>>> 649e19b4 (functioning chatbot shenanigans)

    // Function to send user input to Flask API
    const sendMessage = async () => {
        if (userMessage.trim() === "") return; // Prevent empty messages
        setLoading(true); // Show loading state
        setBotResponse("Thinking..."); // Temporary response

        try {
            const response = await axios.post("http://localhost:5000/generate", {
                user_info: { name: "John", age: 25, school: "GT", feeling: "anxious" },
                emotion: "stressed",
                message: userMessage,
            });

            setBotResponse(response.data.response); // Update with AI response
        } catch (error) {
            console.error("Error fetching response:", error);
            setBotResponse("Oops! Something went wrong.");
        } finally {
            setLoading(false); // Reset loading state
        }
    };

    return (
        <div className="main-page">
            <div className="avatar-box">
                <img src={avatar} alt="TheraBot" className="avatar" />
            </div>
            <div className="chat-box">
                <p>{botResponse}</p>
            </div>
            {/* Input field for user messages */}
            <input
                type="text"
                placeholder="Type your message..."
                value={userMessage}
                onChange={(e) => setUserMessage(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()} // Send on Enter key
                disabled={loading}
                className="chat-input"
            />
            {/* Send button */}
            <button onClick={sendMessage} disabled={loading} className="send-btn">
                {loading ? "Loading..." : "Send"}
            </button>
        </div>
    );
};

export default MainPage;
