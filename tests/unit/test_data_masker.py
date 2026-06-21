from agent_runtime.safety import mask_dict, mask_string


class TestMaskString:
    def test_mask_email(self):
        result = mask_string("Contact: john.doe@example.com for help")
        assert "john.doe@example.com" not in result
        assert "***@***" in result

    def test_mask_phone(self):
        result = mask_string("Call me at 555-123-4567")
        assert "555-123-4567" not in result
        assert "4567" in result  # last 4 preserved

    def test_mask_ssn(self):
        result = mask_string("SSN: 123-45-6789")
        assert "123-45-6789" not in result
        assert "6789" in result

    def test_mask_credit_card(self):
        result = mask_string("Card: 4111-1111-1111-1234")
        assert "4111-1111-1111-1234" not in result
        assert "1234" in result

    def test_mask_password_field(self):
        result = mask_string("password=MySecretPass123")
        assert "MySecretPass123" not in result
        assert "********" in result

    def test_mask_otp(self):
        result = mask_string("Your OTP: 123456 now")
        assert "123456" not in result
        assert "[OTP REDACTED]" in result

    def test_mask_api_key(self):
        result = mask_string("key=sk-abcdefghij0123456789xyz used")
        assert "sk-abcdefghij0123456789xyz" not in result
        assert "[API_KEY_REDACTED]" in result

    def test_empty_string(self):
        assert mask_string("") == ""

    def test_no_sensitive_data(self):
        text = "Hello, how can I help you today?"
        assert mask_string(text) == text

    def test_selective_patterns(self):
        text = "Email: test@example.com Phone: 555-123-4567"
        result = mask_string(text, patterns=["email"])
        assert "test@example.com" not in result
        assert "555-123-4567" in result  # phone not masked

    def test_empty_patterns_list_masks_all(self):
        # Documented verbatim quirk: an explicit empty list is falsy -> falls through
        # to ALL patterns (ithelpdesk-compatible `patterns or list(PATTERNS)`).
        result = mask_string("reach me at a@b.com", patterns=[])
        assert "a@b.com" not in result


class TestMaskDict:
    def test_sensitive_keys_fully_masked(self):
        data = {
            "username": "jdoe",
            "password": "secret123",
            "api_key": "sk-12345",
            "message": "Hello",
        }
        result = mask_dict(data)
        assert result["password"] == "********"
        assert result["api_key"] == "********"
        assert result["username"] == "jdoe"
        assert result["message"] == "Hello"

    def test_nested_dict(self):
        data = {"config": {"token": "abc123", "name": "test"}}
        result = mask_dict(data)
        assert result["config"]["token"] == "********"
        assert result["config"]["name"] == "test"

    def test_list_of_dicts(self):
        data = {"items": [{"secret": "hidden", "name": "item1"}]}
        result = mask_dict(data)
        assert result["items"][0]["secret"] == "********"
        assert result["items"][0]["name"] == "item1"

    def test_string_values_scanned(self):
        data = {"note": "Contact john@example.com for details"}
        result = mask_dict(data)
        assert "john@example.com" not in result["note"]

    def test_non_string_values_passthrough(self):
        data = {"count": 42, "ok": True, "ratio": 1.5}
        result = mask_dict(data)
        assert result == {"count": 42, "ok": True, "ratio": 1.5}

    def test_non_string_keys_do_not_raise(self):
        # SEC-4: int/float/None keys must not raise AttributeError on key.lower().
        result = mask_dict({1: "x", 2.0: "y", None: "z"})
        assert result == {1: "x", 2.0: "y", None: "z"}

    def test_non_string_key_still_matches_sensitive_substring(self):
        # A non-str key is coerced via str() before substring matching; this one
        # does not look sensitive, so its string value is still scanned.
        result = mask_dict({1: "Contact a@b.com"})
        assert "a@b.com" not in result[1]

    def test_deeply_nested_does_not_recurse_unbounded(self):
        # SEC-4: pathologically deep nesting returns without RecursionError.
        data: dict = {"leaf": "ok"}
        for _ in range(500):
            data = {"nested": data}
        result = mask_dict(data)  # must not raise
        assert isinstance(result, dict)
