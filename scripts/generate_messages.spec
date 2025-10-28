# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

# Get the path to the project root (parent of scripts/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))

a = Analysis(
    [os.path.join(project_root, 'src', 'utils', 'generate_messages.py')],
    pathex=[
        os.path.join(project_root, 'src'),
        project_root
    ],
    binaries=[],
    datas=[
        (os.path.join(project_root, 'data', 'datasets', 'mimic-demo-dataset.csv'), 'data/datasets/'),
        (os.path.join(project_root, 'src', 'models', 'simple_gan.pt'), 'models/'),
    ],
    hiddenimports=[
        'torch',
        'numpy', 
        'pandas',
        'models.ehr_gan'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='generate_messages',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='generate_messages'
)