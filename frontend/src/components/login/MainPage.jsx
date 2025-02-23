import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom"; // Import useLocation
import axios from "axios";
import "./mainpage.css"; 
import avatar1 from "./cat1.png";
import avatar2 from "./cat2.png";
import { useNavigate } from "react-router-dom";
<<<<<<< HEAD

=======
>>>>>>> a74cf43b (added flag functionality)

const MainPage = () => {
    const location = useLocation();
    const userInfo = location.state || {}; // Get user info from Login page

    const [avatar, setAvatar] = useState(avatar1);
    const [botResponse, setBotResponse] = useState("Hello, how are you today?");
<<<<<<< HEAD
    const [userMessage, setUserMessage] = useState("");
    const [loading, setLoading] = useState(false);
=======
    const [userMessage, setUserMessage] = useState(""); // User input
    const [loading, setLoading] = useState(false); // Loading state
>>>>>>> a74cf43b (added flag functionality)
    const [isTalking, setIsTalking] = useState(false); // Loading state
    const navigate = useNavigate(); // Loading state

    // Avatar animation (blinking effect)
    useEffect(() => {
        let interval;
        if (!isTalking){
            interval = setInterval(() => {
                setAvatar((prev) => (prev === avatar1 ? avatar2 : avatar1));
            }, 200);
        }else{
            setAvatar(avatar1);
        }
        return () => clearInterval(interval);
<<<<<<< HEAD
    }, []);
=======
    }, [isTalking]);

    // Function to send user input to Flask API
>>>>>>> a74cf43b (added flag functionality)
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
    const handleTalk = () => {
        setIsTalking((prev) => !prev);
    };
    const handleStop = () => {
        navigate("/");
    };
<<<<<<< HEAD

=======
>>>>>>> a74cf43b (added flag functionality)
    return (
        <div className="main-page">
            <div className="avatar-box">
                <img src={avatar} alt="TheraBot" className="avatar" />
            </div>
            <div className="chat-box">
                <p>{botResponse}</p>
            </div>
            <div className="button-box">
                <button className ="talk-toggle" onClick = {handleTalk}> {isTalking ? "Stop" : "Talk"} </button>
                <button className = "stop-btn" onClick = {handleStop}> End Session </button>
            </div>
        </div>
    );
};

export default MainPage;
