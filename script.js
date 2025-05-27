document.addEventListener('DOMContentLoaded', () => {
    const chatLog = document.getElementById('chat-log');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const iterationChoice = document.getElementById('iteration-choice');
    const iterationDetailsDiv = document.getElementById('iteration-details');

    let clientConversationHistory = []; 

    const FLASK_SERVER_URL = 'http://127.0.0.1:5016'; // Define the server URL

    const iterationInfo = {
        v1: {
            name: "Iteration 1: Text-Centric RAG with Basic Image Linking",
            features: [
                "Focus on product descriptions, specifications, and customer reviews.",
                "Images embedded with CLIP; single basic BLIP caption per image used as a textual proxy.",
                "Text retrieval: Bi-encoder similarity search + Cross-encoder reranking.",
                "Image retrieval: CLIP text-to-image, using the single BLIP caption.",
                "LLM Context: Top text chunks, single BLIP captions, basic product metadata.",
                "Simple query parser, no explicit conversation history in LLM prompts (handled by server if at all)."
            ],
            limitations: [
                "Single caption misses key visual details, leading to vague LLM responses.",
                "Weak image contribution; relevance heavily relied on surface-level caption similarity.",
                "Fragmented context: No unified way to treat product text and image-derived text for retrieval.",
                "Limited ability to cross-link images with specific textual claims."
            ]
        },
        v2: {
            name: "Iteration 2: Enhanced Image Captioning and ViLT-Based Reranking",
            features: [
                "Multiple (3-5) diverse BLIP captions generated per image for richer textual representation.",
                "ViLT-based image reranking: Concatenated multiple BLIP captions + user query for precise multimodal alignment.",
                "LLM Context: Includes multiple, diverse BLIP captions for richer grounding.",
                "Conversation history formatted and included in LLM prompts for query parsing and answer generation.",
                "More detailed query parser (e.g., `rewritten_query_for_retrieval`)."
            ],
            limitations: [
                "Still no actual visual region grounding from images themselves or OCR text extraction.",
                "LLM hallucinations could persist with vague visual descriptions if not covered in diverse captions.",
                "Text and image retrieval pipelines remained largely separate, hindering deeper integration."
            ]
        },
        v3: {
            name: "Iteration 3: Deep Multimodal Integration with OCR and Unified Textual Knowledge",
            features: [
                "OCR text extracted from images (e.g., branding, packaging features).",
                "Combined & Filtered Image-Derived Texts: BLIP captions + cleaned OCR text semantically filtered against product metadata (CapFilt-inspired).",
                "Unified Text Index: Filtered, source-attributed image-derived texts (BLIP/OCR) ingested into the *same text Pinecone index* as standard product text.",
                "Direct Retrieval of Image Text: Text retriever can now directly surface relevant BLIP or OCR chunks if they match query semantics.",
                "LLM Context: Includes standard product text AND retrieved image-derived text chunks with clear labeling of the source (e.g., \"Text from image OCR:\", \"AI caption for image:\").",
                "Robust conversation history handling and more resilient JSON parsing from LLM.",
                "Prompts were adapted to instruct Gemini to attribute facts to image text, BLIP captions, or OCR when appropriate."
            ],
            improvements_over_v2: [
                "Image-derived information became semantically retrievable via direct text search.",
                "Combined and filtered BLIP/OCR representations provided more diverse, reliable, and complementary insights from images.",
                "The LLM could explicitly cite visual captions and embedded image text, enriching answer detail and trustworthiness.",
                "Overall grounding and source-aware answer generation were significantly enhanced."
            ]
        }
    };

    function displayIterationDetails(version) {
        const details = iterationInfo[version];
        if (!details) {
            iterationDetailsDiv.innerHTML = "<p>Details for this iteration are not available.</p>";
            return;
        }
        let html = `<h3>${details.name}</h3>`;
        if (details.features && details.features.length > 0) {
            html += `<h4>Key Features:</h4><ul>${details.features.map(f => `<li>${f}</li>`).join('')}</ul>`;
        }
        if (details.limitations && details.limitations.length > 0) {
            html += `<h4>Limitations in this Iteration:</h4><ul>${details.limitations.map(l => `<li>${l}</li>`).join('')}</ul>`;
        }
        if (details.improvements_over_v2 && details.improvements_over_v2.length > 0) {
            html += `<h4>Improvements (over V2):</h4><ul>${details.improvements_over_v2.map(i => `<li>${i}</li>`).join('')}</ul>`;
        }
        iterationDetailsDiv.innerHTML = html;
    }
    
    function appendMessage(sender, messageText, isError = false, isThinking = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        if (isError) {
            messageDiv.classList.add('error-message');
        }
        if (isThinking) {
            messageDiv.classList.add('thinking-message');
        }
        const textNode = document.createTextNode(messageText);
        messageDiv.appendChild(textNode);
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight; 
        return messageDiv; 
    }

    async function checkServerStatusAndEnableInput() {
        userInput.disabled = false;
        sendButton.disabled = false;
        console.log("Chat input enabled. Assuming server is ready.");
    }

    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        appendMessage('user', query);
        userInput.value = '';
        userInput.disabled = true;
        sendButton.disabled = true;
        
        const thinkingDiv = appendMessage('bot', "Thinking...", false, true);

        try {
            // UPDATED FETCH URL
            const response = await fetch(`${FLASK_SERVER_URL}/api/chat`, { 
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    history: clientConversationHistory, 
                    iteration: iterationChoice.value
                }),
            });

            if (thinkingDiv) thinkingDiv.remove();

            if (!response.ok) {
                let errorMsg = `Server error: ${response.status} ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || (errorData.details ? `${errorMsg} - ${errorData.details}`: errorMsg);
                } catch (e) { /* Ignore if error response isn't JSON */ }
                throw new Error(errorMsg);
            }

            const data = await response.json();
            appendMessage('bot', data.answer);
            
            clientConversationHistory = data.updated_history || (clientConversationHistory + [[query, data.answer]]);
            if (clientConversationHistory.length > 7) { 
                 clientConversationHistory = clientConversationHistory.slice(-7);
            }

        } catch (error) {
            console.error('Error sending message:', error);
            if (thinkingDiv) thinkingDiv.remove(); 
            appendMessage('bot', `Error: ${error.message}`, true);
        } finally {
            userInput.disabled = false; 
            sendButton.disabled = false;
            userInput.focus(); 
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    iterationChoice.addEventListener('change', (e) => {
        displayIterationDetails(e.target.value);
    });

    displayIterationDetails(iterationChoice.value);
    checkServerStatusAndEnableInput();
});