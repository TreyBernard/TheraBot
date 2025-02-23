import React, { useState, useEffect } from "react";
import axios from "axios";
import "./mainpage.css"; 
import avatar1 from "./cat1.png";
import avatar2 from "./cat2.png";

const MainPage = () => {
    const [avatar, setAvatar] = useState(avatar1);
    const [botResponse, setBotResponse] = useState("Hello, how are you today?");

  useEffect(() => {
    const interval = setInterval(() => {
        setAvatar((prev) => prev === avatar1 ? avatar2 : avatar1);
}, 150);
    return () => clearInterval(interval);
},[]);

    return (
        <div className="main-page">
            <div className="avatar-box">
                <img src={avatar} alt="TheraBot" className="avatar" />
            </div>
            <div className = "chat-box">
                <p>{botResponse}</p>
            </div>
            
        </div>
    );
};

export default MainPage;
