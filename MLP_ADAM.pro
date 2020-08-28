TEMPLATE = subdirs

CONFIG += precompile_header

SUBDIRS = libmlp    \
    echo/echo_mlp_train_sgd_lr_full \
    echo/echo_train_adam \
    echo/echo_train_amsgrad_full \
    echo_train_sgd  \
    echo_test
    echo_train_adam
    echo_mlp_train_sgd_lr_full

echo_train_sgd.subdir  = echo/echo_train_sgd
echo_test.subdir  = echo/echo_test
echo_train_adam.subdir  = echo/echo_train_adam
echo_mlp_train_sgd_lr_full.subdir  = echo/echo_mlp_train_sgd_lr_full

echo_train_adam.depends  = libmlp
echo_train_sgd.depends  = libmlp
echo_test.depends  = libmlp
echo_mlp_train_sgd_lr_full.depends = libmlp

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/libmlp/release/ -llibmlp
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/libmlp/debug/ -llibmlp
else:unix: LIBS += -L$$OUT_PWD/libmlp/ -llibmlp

INCLUDEPATH += $$PWD/libmlp
DEPENDPATH += $$PWD/libmlp

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/libmlp/release/liblibmlp.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/libmlp/debug/liblibmlp.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/libmlp/release/libmlp.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/libmlp/debug/libmlp.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/libmlp/liblibmlp.a
