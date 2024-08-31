import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Login from "./pages/Login";
import SignUp from "./pages/SignUp";
import Verification from "./pages/Verification";
import Home from "./pages/Home";
import Result from "./pages/Result";

function App() {
  return (
    <>
      <BrowserRouter>
        <div className="App">
          <Routes>
            <Route path="/" element={<Login />} />
            <Route path="/sign_up" element={<SignUp />} />
            <Route path="/sign_up/verify" element={<Verification />} />
            <Route path="/home" element={<Home />} />
            <Route path="/result" element={<Result />} />
          </Routes>
        </div>
      </BrowserRouter>
    </>
  );
}

export default App;
