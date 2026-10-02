import pytest
from unittest.mock import MagicMock, patch
from friday import config
from friday.config import TTSConfig
from friday.tts import TextToSpeech, _Pyttsx3Backend, TTSStatus

def test_tts_setup_order():
    # We want to test that TextToSpeech.setup() prioritizes:
    # Kokoro -> EdgeTTS -> pyttsx3
    
    mock_tts = TTSConfig(tts_engine="kokoro")

    with patch("friday.tts._KOKORO_AVAILABLE", True), \
         patch("friday.tts._EDGE_TTS_AVAILABLE", True), \
         patch("friday.tts._PYTTSX3_AVAILABLE", True), \
         patch.object(config, "tts", mock_tts):
         
        # Mock each backend class's setup method
        with patch("friday.tts._KokoroBackend") as MockKokoro, \
             patch("friday.tts._EdgeTTSBackend") as MockEdge, \
             patch("friday.tts._Pyttsx3Backend") as MockPyttsx3:
             
            # Create instances and configure mock setup
            kokoro_inst = MockKokoro.return_value
            edge_inst = MockEdge.return_value
            pyttsx3_inst = MockPyttsx3.return_value
            
            # Make names return something identifiable
            kokoro_inst.name = "Kokoro"
            edge_inst.name = "EdgeTTS"
            pyttsx3_inst.name = "pyttsx3"
            
            tts = TextToSpeech()
            success = tts.setup()
            
            assert success is True
            assert len(tts._fallback_chain) == 3
            assert tts._fallback_chain[0].name == "Kokoro"
            assert tts._fallback_chain[1].name == "EdgeTTS"
            assert tts._fallback_chain[2].name == "pyttsx3"

def test_pyttsx3_backend_com_initialization():
    with patch("friday.tts._PYTTSX3_AVAILABLE", True), \
         patch("friday.tts._PYTHONCOM_AVAILABLE", True), \
         patch("friday.tts.pythoncom.CoInitialize") as mock_coinit, \
         patch("friday.tts.pyttsx3.init") as mock_pyttsx3_init:
         
        mock_engine = MagicMock()
        mock_pyttsx3_init.return_value = mock_engine
        mock_engine.getProperty.return_value = []
        
        backend = _Pyttsx3Backend()
        backend.setup()
        
        # COM and the engine are initialized on the dedicated worker thread.
        mock_coinit.assert_called_once()
        mock_pyttsx3_init.assert_called_once()
