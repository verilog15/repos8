import operator
import os
from base64 import urlsafe_b64encode
from datetime import datetime, timedelta, timezone
from typing import Union

import store
import werkzeug
from db import db
from encryption import EncryptionManager, GpgDecryptError, GpgKeyNotFoundError
from flask import (
    Blueprint,
    abort,
    current_app,
    g,
    make_response,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_babel import gettext
from models import InstanceConfig, Reply, Submission, get_one_or_else
from passphrases import DicewarePassphrase, PassphraseGenerator
from sdconfig import SecureDropConfig
from source_app.decorators import login_required
from source_app.forms import LoginForm, SubmissionForm
from source_app.session_manager import SessionManager
from source_app.utils import (
    clear_session_and_redirect_to_logged_out_page,
    codename_detected,
    fit_codenames_into_cookie,
    flash_msg,
    normalize_timestamps,
)
from source_user import (
    InvalidPassphraseError,
    SourceDesignationCollisionError,
    SourcePassphraseCollisionError,
    SourceUser,
    create_source_user,
)
from store import Storage

import redwood
from redwood import RedwoodError


def make_blueprint(config: SecureDropConfig) -> Blueprint:
    view = Blueprint("main", __name__)

    @view.route("/")
    def index() -> str:
        return render_template("index.html")

    @view.route("/generate", methods=("POST", "GET"))
    def generate() -> Union[str, werkzeug.Response]:
        if request.method == "POST":
            # Try to detect Tor2Web usage by looking to see if tor2web_check got mangled
            tor2web_check = request.form.get("tor2web_check")
            if tor2web_check is None:
                # Missing form field
                abort(403)
            elif tor2web_check != 'href="fake.onion"':
                return redirect(url_for("info.tor2web_warning"))

        if SessionManager.is_user_logged_in(db_session=db.session):
            flash_msg(
                "notification",
                None,
                gettext(
                    "You were redirected because you are already logged in. "
                    "If you want to create a new account, you should log out first."
                ),
            )
            return redirect(url_for(".lookup"))
        codename = PassphraseGenerator.get_default().generate_passphrase(
            preferred_language=g.localeinfo.language
        )

        # Generate a unique id for each browser tab and associate the codename with this id.
        # This will allow retrieval of the codename displayed in the tab from which the source has
        # clicked to proceed to /generate (ref. issue #4458)
        tab_id = urlsafe_b64encode(os.urandom(64)).decode()
        codenames = session.get("codenames", {})
        codenames[tab_id] = codename
        session["codenames"] = fit_codenames_into_cookie(codenames)
        session["codenames_expire"] = datetime.now(timezone.utc) + timedelta(
            minutes=config.SESSION_EXPIRATION_MINUTES
        )
        return render_template("generate.html", codename=codename, tab_id=tab_id)

    @view.route("/create", methods=["POST"])
    def create() -> werkzeug.Response:
        if SessionManager.is_user_logged_in(db_session=db.session):
            flash_msg(
                "notification",
                None,
                gettext(
                    "You are already logged in. Please verify your codename as it "
                    "may differ from the one displayed on the previous page."
                ),
            )
        else:
            # Ensure the codenames have not expired
            date_codenames_expire = session.get("codenames_expire")
            if not date_codenames_expire or datetime.now(timezone.utc) >= date_codenames_expire:
                return clear_session_and_redirect_to_logged_out_page(flask_session=session)

            tab_id = request.form["tab_id"]
            codename = session["codenames"][tab_id]
            del session["codenames"]

            try:
                current_app.logger.info("Creating new source user...")
                create_source_user(
                    db_session=db.session,
                    source_passphrase=codename,
                    source_app_storage=Storage.get_default(),
                )
            except (SourcePassphraseCollisionError, SourceDesignationCollisionError) as e:
                current_app.logger.error(f"Could not create a source: {e}")
                flash_msg(
                    "error",
                    None,
                    gettext(
                        "There was a temporary problem creating your account. Please try again."
                    ),
                )
                return redirect(url_for(".index"))

            # All done - source user was successfully created
            current_app.logger.info("New source user created")
            session["new_user_codename"] = codename
            SessionManager.log_user_in(
                db_session=db.session, supplied_passphrase=DicewarePassphrase(codename)
            )

        return redirect(url_for(".lookup"))

    @view.route("/lookup", methods=("GET",))
    @login_required
    def lookup(logged_in_source: SourceUser) -> str:
        replies = []
        logged_in_source_in_db = logged_in_source.get_db_record()
        source_inbox = Reply.query.filter_by(
            source_id=logged_in_source_in_db.id, deleted_by_source=False
        ).all()

        first_submission = logged_in_source_in_db.interaction_count == 0

        if first_submission:
            min_message_length = InstanceConfig.get_default().initial_message_min_len
        else:
            min_message_length = 0

        for reply in source_inbox:
            reply_path = Storage.get_default().path(
                logged_in_source.filesystem_id,
                reply.filename,
            )
            try:
                with open(reply_path, "rb") as f:
                    contents = f.read()
                decrypted_reply = EncryptionManager.get_default().decrypt_journalist_reply(
                    for_source_user=logged_in_source,
                    ciphertext_in=contents,
                )
                reply.decrypted = decrypted_reply
            except UnicodeDecodeError:
                current_app.logger.error(f"Could not decode reply {reply.filename}")
            except FileNotFoundError:
                current_app.logger.error(f"Reply file missing: {reply.filename}")
            except (GpgDecryptError, RedwoodError) as e:
                current_app.logger.error(f"Could not decrypt reply {reply.filename}: {str(e)}")
            else:
                reply.date = datetime.utcfromtimestamp(os.stat(reply_path).st_mtime)
                replies.append(reply)

        # Sort the replies by date
        replies.sort(key=operator.attrgetter("date"), reverse=True)

        return render_template(
            "lookup.html",
            is_user_logged_in=True,
            allow_document_uploads=InstanceConfig.get_default().allow_document_uploads,
            replies=replies,
            min_len=min_message_length,
            new_user_codename=session.get("new_user_codename", None),
            form=SubmissionForm(),
        )

    @view.route("/submit", methods=("POST",))
    @login_required
    def submit(logged_in_source: SourceUser) -> werkzeug.Response:
        allow_document_uploads = InstanceConfig.get_default().allow_document_uploads
        form = SubmissionForm()
        if not form.validate():
            for errors in form.errors.values():
                for error in errors:
                    flash_msg("error", None, error)
            return redirect(url_for("main.lookup"))

        msg = request.form["msg"]
        fh = None
        if allow_document_uploads and "fh" in request.files:
            fh = request.files["fh"]

        # Don't submit anything if it was an "empty" submission. #878
        if not (msg or fh):
            if allow_document_uploads:
                html_contents = gettext("You must enter a message or choose a file to submit.")
            else:
                html_contents = gettext("You must enter a message.")

            flash_msg("error", None, html_contents)
            return redirect(url_for("main.lookup"))

        fnames = []
        logged_in_source_in_db = logged_in_source.get_db_record()
        first_submission = logged_in_source_in_db.interaction_count == 0

        if first_submission:
            min_len = InstanceConfig.get_default().initial_message_min_len
            if (min_len > 0) and (msg and not fh) and (len(msg) < min_len):
                flash_msg(
                    "error",
                    None,
                    gettext("Your first message must be at least {} characters long.").format(
                        min_len
                    ),
                )
                return redirect(url_for("main.lookup"))

            # if the new_user_codename key is not present in the session, this is
            # not a first session
            new_codename = session.get("new_user_codename", None)

            codenames_rejected = InstanceConfig.get_default().reject_message_with_codename
            if new_codename is not None:
                if codenames_rejected and codename_detected(msg, new_codename):
                    flash_msg(
                        "error",
                        None,
                        gettext("Please do not submit your codename!"),
                        gettext(
                            "Keep your codename secret, and use it to log in later to "
                            "check for replies."
                        ),
                    )
                    return redirect(url_for("main.lookup"))

        if not os.path.exists(Storage.get_default().path(logged_in_source.filesystem_id)):
            current_app.logger.debug(
                f"Store directory not found for source "
                f"'{logged_in_source_in_db.journalist_designation}', creating one."
            )
            os.mkdir(Storage.get_default().path(logged_in_source.filesystem_id))

        if msg:
            logged_in_source_in_db.interaction_count += 1
            fnames.append(
                Storage.get_default().save_message_submission(
                    logged_in_source_in_db.filesystem_id,
                    logged_in_source_in_db.interaction_count,
                    logged_in_source_in_db.journalist_filename,
                    msg,
                )
            )
        if fh:
            logged_in_source_in_db.interaction_count += 1
            fnames.append(
                Storage.get_default().save_file_submission(
                    logged_in_source_in_db.filesystem_id,
                    logged_in_source_in_db.interaction_count,
                    logged_in_source_in_db.journalist_filename,
                    fh.filename,
                    fh.stream,
                )
            )

        if first_submission or msg or fh:
            if first_submission:
                html_contents = gettext(
                    "Thank you for sending this information to us. Please "
                    "check back later for replies."
                )
            elif msg and not fh:
                html_contents = gettext("Thanks! We received your message.")
            elif fh and not msg:
                html_contents = gettext("Thanks! We received your file.")
            else:
                html_contents = gettext("Thanks! We received your file and message.")

            flash_msg("success", gettext("Success!"), html_contents)

        new_submissions = []
        for fname in fnames:
            submission = Submission(logged_in_source_in_db, fname, Storage.get_default())
            db.session.add(submission)
            new_submissions.append(submission)

        logged_in_source_in_db.pending = False
        logged_in_source_in_db.last_updated = datetime.now(timezone.utc)
        db.session.commit()

        for sub in new_submissions:
            store.async_add_checksum_for_file(sub, Storage.get_default())

        normalize_timestamps(logged_in_source)

        return redirect(url_for("main.lookup"))

    @view.route("/delete", methods=("POST",))
    @login_required
    def delete(logged_in_source: SourceUser) -> werkzeug.Response:
        """This deletes the reply from the source's inbox, but preserves
        the history for journalists such that they can view conversation
        history.
        """

        query = Reply.query.filter_by(
            filename=request.form["reply_filename"], source_id=logged_in_source.db_record_id
        )
        reply = get_one_or_else(query, current_app.logger, abort)
        reply.deleted_by_source = True
        db.session.add(reply)
        db.session.commit()

        flash_msg("success", gettext("Success!"), gettext("Reply deleted"))
        return redirect(url_for(".lookup"))

    @view.route("/delete-all", methods=("POST",))
    @login_required
    def batch_delete(logged_in_source: SourceUser) -> werkzeug.Response:
        replies = (
            Reply.query.filter(Reply.source_id == logged_in_source.db_record_id)
            .filter(Reply.deleted_by_source == False)
            .all()
        )
        if len(replies) == 0:
            current_app.logger.error("Found no replies when at least one was " "expected")
            return redirect(url_for(".lookup"))

        for reply in replies:
            reply.deleted_by_source = True
            db.session.add(reply)
        db.session.commit()

        flash_msg("success", gettext("Success!"), gettext("All replies have been deleted"))
        return redirect(url_for(".lookup"))

    @view.route("/login", methods=("GET", "POST"))
    def login() -> Union[str, werkzeug.Response]:
        form = LoginForm()
        if not form.validate_on_submit():
            return render_template("login.html", form=form)
        try:
            source_user = SessionManager.log_user_in(
                db_session=db.session,
                supplied_passphrase=DicewarePassphrase(request.form["codename"].strip()),
            )
        except InvalidPassphraseError:
            current_app.logger.info("Login failed for invalid codename")
            flash_msg("error", None, gettext("Sorry, that is not a recognized codename."))
            return render_template("login.html", form=form)
        # Success: a valid passphrase was supplied and the source was logged-in
        source = source_user.get_db_record()
        if source.fingerprint is None:
            # This legacy source didn't have a PGP keypair generated yet,
            # do it now.
            public_key, secret_key, fingerprint = redwood.generate_source_key_pair(
                source_user.gpg_secret, source_user.filesystem_id
            )
            source.pgp_public_key = public_key
            source.pgp_secret_key = secret_key
            source.pgp_fingerprint = fingerprint
            db.session.add(source)
            db.session.commit()
        elif source.pgp_secret_key is None:
            # Need to migrate the secret key out of GPG
            encryption_mgr = EncryptionManager.get_default()
            try:
                secret_key = encryption_mgr.get_source_secret_key_from_gpg(
                    source.fingerprint, source_user.gpg_secret
                )
            except GpgKeyNotFoundError:
                # Don't fail here, but it's likely decryption of journalist
                # messages will fail.
                secret_key = None
            if secret_key:
                source.pgp_secret_key = secret_key
                db.session.add(source)
                db.session.commit()
                # Let's optimistically delete the GPG key from the keyring
                # if *everything* has been migrated. If this fails it's OK,
                # since we still try to delete from the keyring on source
                # deletion too.
                if source.pgp_fingerprint and source.pgp_public_key:
                    try:
                        encryption_mgr.delete_source_key_pair(source.filesystem_id)
                    except:  # noqa: E722
                        pass

        return redirect(url_for(".lookup", from_login="1"))

    @view.route("/logout", methods=("POST",))
    @login_required
    def logout(logged_in_source: SourceUser) -> Union[str, werkzeug.Response]:
        """
        If a user is logged in, show them a logout page that prompts them to
        click the New Identity button in Tor Browser to complete their session.
        Otherwise redirect to the main Source Interface page.
        """
        if SessionManager.is_user_logged_in(db_session=db.session):
            SessionManager.log_user_out()

            # Clear the session after we render the message so it's localized
            # If a user specified a locale, save it and restore it
            session.clear()
            session["locale"] = g.localeinfo.id

            return render_template("logout.html")
        else:
            return redirect(url_for(".index"))

    @view.route("/robots.txt")
    def robots_txt() -> werkzeug.Response:
        """Tell robots we don't want them"""
        resp = make_response("User-agent: *\nDisallow: /")
        resp.headers["content-type"] = "text/plain"
        return resp

    return view
