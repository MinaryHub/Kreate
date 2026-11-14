package com.minaryhub.kreate;

import android.app.NativeActivity;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Kreate's activity: the NativeActivity the program runs in, and the system's
 * file picker, which a NativeActivity alone cannot open or hear back from.
 *
 * The program (libkreate_android.so) calls {@link #openDocument} and
 * {@link #createDocument} through JNI and goes on; the picker answers here,
 * and the copying is done on a thread of its own:
 *
 * - A file opened is copied into the app's Documents folder, and the program
 *   is told where through {@link #nativeFileArrived}.
 * - A file saved is already in Documents; it is copied to where the person
 *   chose.
 */
public class KreateActivity extends NativeActivity {
    private static final int OPEN = 1;
    private static final int SAVE = 2;

    static {
        // Loaded by NativeActivity as well; loading it here is what lets
        // the VM find nativeFileArrived in it.
        System.loadLibrary("kreate_android");
    }

    /** What is being saved, while the person chooses where. */
    private String saving;

    private static native void nativeFileArrived(String path);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (savedInstanceState != null) {
            saving = savedInstanceState.getString("saving");
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        outState.putString("saving", saving);
    }

    private File documents() {
        File dir = new File(getExternalFilesDir(null), "Documents");
        dir.mkdirs();
        return dir;
    }

    /** The system's picker, for a file to open. */
    public void openDocument() {
        runOnUiThread(() -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            startActivityForResult(intent, OPEN);
        });
    }

    /** The system's save screen, for a copy of {@code path} under {@code name}. */
    public void createDocument(String path, String name) {
        runOnUiThread(() -> {
            saving = path;
            Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("application/octet-stream");
            intent.putExtra(Intent.EXTRA_TITLE, name);
            startActivityForResult(intent, SAVE);
        });
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (result != RESULT_OK || data == null || data.getData() == null) {
            return;
        }
        final Uri uri = data.getData();
        if (request == OPEN) {
            new Thread(() -> bringIn(uri)).start();
        } else if (request == SAVE && saving != null) {
            final String from = saving;
            saving = null;
            new Thread(() -> takeOut(from, uri)).start();
        }
    }

    private String nameOf(Uri uri) {
        String name = null;
        try (Cursor c = getContentResolver().query(uri, null, null, null, null)) {
            if (c != null && c.moveToFirst()) {
                int i = c.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (i >= 0) {
                    name = c.getString(i);
                }
            }
        } catch (Exception ignored) {
        }
        if (name == null || name.isEmpty()) {
            name = "opened";
        }
        // A name, not a path.
        return new File(name).getName();
    }

    private static void copy(InputStream in, OutputStream out) throws java.io.IOException {
        byte[] buffer = new byte[1 << 16];
        int n;
        while ((n = in.read(buffer)) > 0) {
            out.write(buffer, 0, n);
        }
    }

    private void bringIn(Uri uri) {
        File to = new File(documents(), nameOf(uri));
        try (InputStream in = getContentResolver().openInputStream(uri);
             OutputStream out = new FileOutputStream(to)) {
            if (in == null) {
                return;
            }
            copy(in, out);
        } catch (Exception e) {
            say("Could not open the file: " + e.getMessage());
            return;
        }
        nativeFileArrived(to.getPath());
    }

    private void takeOut(String from, Uri uri) {
        try (InputStream in = new FileInputStream(from);
             OutputStream out = getContentResolver().openOutputStream(uri, "wt")) {
            if (out == null) {
                return;
            }
            copy(in, out);
            say("Saved " + new File(from).getName());
        } catch (Exception e) {
            say("Could not save the copy: " + e.getMessage());
        }
    }

    private void say(String text) {
        runOnUiThread(() -> Toast.makeText(this, text, Toast.LENGTH_LONG).show());
    }
}
