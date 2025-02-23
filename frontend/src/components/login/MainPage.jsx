import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom"; // Import useLocation and useNavigate
import axios from "axios";
import "./mainpage.css"; 
import avatar1 from "./cat1.png";
import avatar2 from "./cat2.png";

const MainPage = () => {
    const location = useLocation();
    const userInfo = location.state || {}; // Get user info from Login page
    const navigate = useNavigate();

    const [avatar, setAvatar] = useState(avatar1);
    const [botResponse, setBotResponse] = useState("Hello, how are you today?");
    const [userMessage, setUserMessage] = useState(""); // Stores transcribed text from speech-to-text
    const [loading, setLoading] = useState(false);
    const [isTalking, setIsTalking] = useState(false);
    const [isThinking, setIsThinking] = useState(false); // AI thinking flag
    const [recognition, setRecognition] = useState(null);

    // Avatar animation (blinking effect)
    useEffect(() => {
        let interval;
        if (!isTalking && !isThinking) {
            interval = setInterval(() => {
                setAvatar((prev) => (prev === avatar1 ? avatar2 : avatar1));
            }, 200);
        } else {
            setAvatar(avatar1);
        }
        return () => clearInterval(interval);
    }, [isTalking, isThinking]);

    // Set up Speech Recognition on mount
    useEffect(() => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            console.error("Speech Recognition API not supported in this browser.");
            return;
        }
        const recog = new SpeechRecognition();
        recog.continuous = true;
        recog.interimResults = true;
        recog.lang = "en-US";
        recog.onresult = (event) => {
            let finalTranscript = "";
            let interimTranscript = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcriptChunk = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcriptChunk;
                } else {
                    interimTranscript += transcriptChunk;
                }
            }
            // Update userMessage state with the transcript
            setUserMessage(finalTranscript || interimTranscript);
        };
        recog.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
        };
        setRecognition(recog);
    }, []);

    // Function to send user input to Flask API
    const sendMessage = async () => {
        if (userMessage.trim() === "") return; // Prevent empty messages
        setLoading(true);
        setBotResponse("Thinking...");
        setIsThinking(true);
        try {
            const response = await axios.post("http://localhost:5000/generate", {
                user_info: {
                    name: userInfo.username || "Unknown",
                    school: userInfo.school || "Unknown",
                    age: userInfo.age || "Unknown",
                    feeling: userInfo.feeling || "Unknown",
                },
                message: userMessage,  // Transcribed text stored here
            });
            setBotResponse(response.data.response);
        } catch (error) {
            console.error("Error fetching response:", error);
            setBotResponse("Oops! Something went wrong.");
        } finally {
            setLoading(false);
            setIsThinking(false);
        }
    };

    // Function to update talking state in backend
    const updateTalkingStatus = async (newState) => {
        try {
            await axios.post("http://localhost:5000/update_talking_status", {
                username: userInfo.username || "default_user",
                isTalking: newState,
            });
            console.log(`Sent isTalking=${newState} to backend`);
        } catch (error) {
            console.error("Error updating talking state:", error);
        }
    };

    // Handle Talk Button Click: Toggle speech recognition on/off
    const handleTalk = async () => {
        const newState = !isTalking;
        setIsTalking(newState);
        updateTalkingStatus(newState);
        if (recognition) {
            if (newState) {
                // Start speech recognition when toggled on
                recognition.start();
            } else {
                // Stop speech recognition when toggled off
                recognition.stop();
                // Optionally, send the transcribed text automatically when speech stops
                sendMessage();
            }
        }
    };

    // Handle End Session Button: Stop recognition (if active) and navigate away
    const handleStop = () => {
        if (recognition && isTalking) {
            recognition.stop();
        }
        navigate("/");
    };

    return (
        <div className="main-page">
            <div className="avatar-box">
                <img src={avatar} alt="TheraBot" className="avatar" />
            </div>
            <div className="chat-box">
                <p>{botResponse}</p>
            </div>
            <div className="button-box">
                <button className="talk-toggle" onClick={handleTalk}>
                    {isTalking ? "Stop" : "Talk"}
                </button>
                <button className="stop-btn" onClick={handleStop}>
                    End Session
                </button>
            </div>
            {/* Optional: Display the current transcript */}
            <div className="transcript-box">
                <p>Transcript: {userMessage}</p>
            </div>
        </div>
    );
};

export default MainPage;
