from scripts.check_public_release import inspect_content, prohibited_path


def test_private_paths_remain_excluded():
    for name in ['.env', 'nested/.env.local', 'training/chat.txt', 'data/contact_profiles.json',
                 'data/logs/screenshot.png', 'identity/profile.json', 'backup.db']:
        assert prohibited_path(name)
    assert not prohibited_path('.env.example')
    assert not prohibited_path('data/selectors/screen_signatures.json')


def test_credentials_are_reported_without_the_value():
    credential = 'ghp_' + 'a' * 36
    findings = inspect_content('example.py', credential.encode())
    assert any('GitHub credential' in finding for finding in findings)
    assert all(credential not in finding for finding in findings)


def test_reserved_phone_fixture_allowed_and_other_numbers_flagged():
    assert not inspect_content('tests/fixture.py', b'+44 7700 900123')
    number = '+44 ' + '7123 ' + '456789'
    assert inspect_content('tests/fixture.py', number.encode())


def test_unreviewed_binary_is_rejected():
    assert inspect_content('capture.png', b'PNG\0binary')
    assert not inspect_content('docs/images/demo.png', b'PNG\0binary')
