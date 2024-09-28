<script>
    let chatVisible = false;

    // Toggle the chat visibility
    const toggleChat = () => {
        chatVisible = !chatVisible;
    }
</script>

<style>
    /* Style for the chat bubble */
    #chat-bubble {
        position: fixed;
        bottom: 100px; /* Adjust this value to be above the toggle button */
        right: 30px; /* Adjusted to match the increased size */
        width: 450px; /* Increased by 50% from 300px */
        height: 600px; /* Increased by 50% from 400px */
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.8); /* Slightly increased shadow */
        z-index: 1001; /* Increased z-index to be above the button */
        display: block; /* Show when rendered */
        transition: opacity 0.3s ease, bottom 0.3s ease; /* Smooth transition for appearance and position */
    }

    /* Hide chat bubble by default and show when visible */
    #chat-bubble.hidden {
        display: none; /* Initially hidden */
    }

    /* Style for the iframe to fill the chat bubble */
    #chat-container {
        width: 100%; /* Full width */
        height: 100%; /* Full height */
        border: none; /* Remove default border */
    }

    /* Style for the toggle button */
    #chat-toggle {
        position: fixed;
        bottom: 30px; /* Position the button above the chat bubble */
        right: 30px; /* Adjusted to match the increased size */
        background-color: #ffffff;
        color: black; /* Change text color to black for contrast */
        border: 3px solid black; /* Outline with a color */
        border-radius: 50%;
        width: 60px; /* Adjusted size */
        height: 60px; /* Adjusted size */
        cursor: pointer;
        z-index: 1000; /* Lower z-index so it appears below the chat bubble */
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease; /* Smooth transition for hover effect */
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2); /* Add slight shadow for depth */
    }

    /* Style for the icon */
    #chat-icon {
        width: 30px; /* Smaller icon size */
        height: 30px; /* Smaller icon size */
        transition: transform 0.3s ease; /* Smooth transition for the icon */
    }

    /* Hover effect */
    #chat-toggle:hover {
        background-color: #f74b61; /* Change background on hover */
        color: white; /* Change text color on hover */
        transform: scale(1.1); /* Slightly enlarge button */
    }

    /* Hover effect for icon */
    #chat-toggle:hover #chat-icon {
        transform: scale(1.1); /* Slightly enlarge icon */
    }
</style>

<!-- Button to open/close the chat bubble -->
<button id="chat-toggle" on:click={toggleChat} class:active={chatVisible}>
    <img id="chat-icon" src="icons/icons8-chatbot-48.png" alt="Chatbot Icon" />
</button>

<!-- Chat bubble container -->
<div id="chat-bubble" class:hidden={!chatVisible} style:bottom={chatVisible ? '100px' : '30px'}>
    <iframe id="chat-container" src="http://localhost:8000" title="Chainlit Chat"></iframe>
</div>
