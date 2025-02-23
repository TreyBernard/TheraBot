import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom"; // Import useLocation
import axios from "axios";
import "./mainpage.css"; 
import avatar1 from "./cat1.png";
import avatar2 from "./cat2.png";

const MainPage = () => {
    const location = useLocation();
    const userInfo = location.state || {}; // Get user info from Login page

    const [avatar, setAvatar] = useState(avatar1);
    const [botResponse, setBotResponse] = useState("Hello, how are you today?");
    const [userMessage, setUserMessage] = useState("");
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const interval = setInterval(() => {
            setAvatar((prev) => (prev === avatar1 ? avatar2 : avatar1));
        }, 200);
        return () => clearInterval(interval);
    }, []);

    const sendMessage = async () => {
        if (userMessage.trim() === "") return; // Prevent empty messages

        setLoading(true);
        setBotResponse("Thinking...");

        try {
            const response = await axios.post("http://localhost:5000/generate", {
                user_info: {
                    name: userInfo.username || "Unknown",
                    school: userInfo.school || "Unknown",
                    age: userInfo.age || "Unknown",
                    feeling: userInfo.feeling || "Unknown",
                },
                emotion: "stressed",
                message: userMessage,
            });

            setBotResponse(response.data.response);
        } catch (error) {
            console.error("Error fetching response:", error);
            setBotResponse("Oops! Something went wrong.");
        } finally {
            setLoading(false);
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
            <input
                type="text"
                placeholder="Type your message..."
                value={userMessage}
                onChange={(e) => setUserMessage(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                disabled={loading}
                className="chat-input"
            />
            <button onClick={sendMessage} disabled={loading} className="send-btn">
                {loading ? "Loading..." : "Send"}
            </button>
        </div>
    );
};

export default MainPage;
